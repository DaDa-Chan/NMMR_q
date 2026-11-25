import numpy as np
import pandas as pd
from scipy import optimize
from pathlib import Path

import time
from src.data.ate.sgd_pv import generate_data
from src.utils import make_seed_bundle
from typing import Dict, Any, List, Optional, Sequence, Tuple

Q_FORMS = ("pooled", "arm_specific")


def default_proxy_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Column names used when none are given (the S1 layout): Z = ['Z'] (or
    every column starting with 'Z'), W likewise, X = ['X1', 'X2'] if present,
    otherwise no covariates.
    """
    z_cols = ['Z'] if 'Z' in df.columns else [c for c in df.columns if c.startswith('Z')]
    w_cols = ['W'] if 'W' in df.columns else [c for c in df.columns if c.startswith('W')]
    x_cols = ['X1', 'X2'] if 'X1' in df.columns else []
    return z_cols, w_cols, x_cols


class ProximalLinearModel:
    """
    Parametric proximal bridges of Cui et al. (2024), eqs. (10), (11), (13), (14).

    h(W,A,X;b) = b0 + ba*A + bw'W + bx'X, solved from
        E_n{[Y - h](1, Z, A, X)'} = 0.

    q has two forms:

    * ``q_form="pooled"`` (eq. 14):
        q(Z,A,X;t) = 1 + exp{(-1)^(1-A) (t0 + tz'Z + ta*A + tx'X)},
      solved from E_n{(-1)^(1-A) q (1,W,A,X)' - (0, 0_pw, 1, 0_px)'} = 0.
    * ``q_form="arm_specific"``: eq. (14) with all Z-A and X-A interaction
      terms (the specification of Cui et al.'s RHC analysis). With full
      interactions the model is one free model per arm,
        q(Z,a,X) = 1 + exp(theta_a'(1, Z, X)),
      and with n(W,A,X) extended by (W*A, X*A) the estimating equations
      decouple into, for a = 0, 1,
        E_n{1(A=a) q(Z,a,X) (1,W,X)' - (1,W,X)'} = 0.
      Requires dim(Z) == dim(W). ``t`` is (theta_0, theta_1) concatenated.

    The default column layout (no column arguments, ``q_form="pooled"``) is
    the S1 layout and is numerically unchanged from the original class.
    """

    def __init__(self, df, z_cols: Optional[Sequence[str]] = None,
                 w_cols: Optional[Sequence[str]] = None,
                 x_cols: Optional[Sequence[str]] = None,
                 q_form: str = "pooled"):
        """
        df 必须含 A、Y；Z/W/X 的列名可显式给出，缺省时按 S1 的列名推断
        （见 default_proxy_columns）。
        """
        if q_form not in Q_FORMS:
            raise ValueError(f"q_form must be one of {Q_FORMS}; got {q_form!r}.")
        self.n = len(df)
        self.q_form = q_form

        default_z, default_w, default_x = default_proxy_columns(df)
        self.z_cols = list(default_z if z_cols is None else z_cols)
        self.w_cols = list(default_w if w_cols is None else w_cols)
        self.x_cols = list(default_x if x_cols is None else x_cols)

        self.A = df['A'].values
        self.Y = df['Y'].values
        self.Z = df[self.z_cols].values
        self.W = df[self.w_cols].values
        self.X = df[self.x_cols].values if self.x_cols else np.zeros((self.n, 0))

        if q_form == "arm_specific" and self.Z.shape[1] != self.W.shape[1]:
            raise ValueError(
                "q_form='arm_specific' needs dim(Z) == dim(W) for an exactly identified "
                f"system; got dim(Z)={self.Z.shape[1]}, dim(W)={self.W.shape[1]}."
            )

        self.ones = np.ones((self.n, 1))

    def arrays(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(W, Z, X) of any frame with this model's columns (fold eval rows, test set)."""
        X = df[self.x_cols].values if self.x_cols else np.zeros((len(df), 0))
        return df[self.w_cols].values, df[self.z_cols].values, X

    @property
    def dim_b(self) -> int:
        return 2 + self.W.shape[1] + self.X.shape[1]

    @property
    def dim_t(self) -> int:
        if self.q_form == "pooled":
            return 2 + self.Z.shape[1] + self.X.shape[1]
        return 2 * (1 + self.Z.shape[1] + self.X.shape[1])

    def h_func(self, b, W, A, X):
        """
        Outcome Bridge Function h(W, A, X; b)
        h(W,A,X;b) = b0 + ba*A + bw*W + bx*X

        W/A/X may belong to ANY sample set (this fold's train rows, its eval
        rows, or the independent test set) -- the row count is taken from
        the arguments themselves, not from `self.n` (the fold this model
        was fit on), so this is safe to call for cross-fitting prediction.
        """
        A_mat = A.reshape(-1, 1)
        ones = np.ones((A_mat.shape[0], 1))
        feats = np.hstack([ones, A_mat, W, X])
        return feats @ b

    def q_func(self, t, Z, A, X):
        """
        Treatment Bridge Function q(Z, A, X; t)
        q(...) = 1 + exp{ (-1)^(1-A) * [t0 + tz*Z + ta*A + tx*X] }

        Same row-count independence as `h_func` -- see its docstring.
        For q_form="arm_specific": q = 1 + exp(theta_A'(1, Z, X)).
        """
        if self.q_form == "arm_specific":
            return self._q_arm_specific(t, Z, A, X)[0]
        A_mat = A.reshape(-1, 1)
        ones = np.ones((A_mat.shape[0], 1))

        feats = np.hstack([ones, Z, A_mat, X])
        linear_part = feats @ t

        sign = (2 * A - 1)
        return 1.0 + np.exp(sign * linear_part)

    def solve_h(self):
        """
        求解 h 的参数 b
        Estimating Equation: E_n { [Y - h(W,A,X;b)] * (1, Z, A, X)^T } = 0

        Returns (b_opt, used_lstsq): `used_lstsq` records whether the exact
        solve was singular and the lstsq fallback was used. It is a
        diagnostic only; a large condition number is not treated as failure.
        """
        A_mat = self.A.reshape(-1, 1)
        M = np.hstack([self.ones, self.Z, A_mat, self.X])

        V = np.hstack([self.ones, A_mat, self.W, self.X])

        dim_b = V.shape[1]
        dim_eq = M.shape[1]

        if dim_b != dim_eq:
            print(f"Warning: h parameters ({dim_b}) != equations ({dim_eq}). System may be over/under-determined.")

        lhs = M.T @ V
        rhs = M.T @ self.Y
        used_lstsq = False
        try:
            b_opt = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            b_opt = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            used_lstsq = True

        return b_opt, used_lstsq

    def solve_q(self):
        """
        求解 q 的参数 t
        Estimating Equation:
        E_n { (-1)^(1-A) * q(Z,A,X;t) * (1,W,A,X)^T - (0, 0_pw, 1, 0_px)^T } = 0

        Returns (t_opt, converged, message, residual_norm): `converged` is
        the root finder's own success flag (never silently treated as a
        pass); `residual_norm` is the final moment-equation residual's L2
        norm, saved as a diagnostic only -- never a pass/fail threshold.
        For q_form="arm_specific" see `_solve_q_arm_specific`.
        """
        if self.q_form == "arm_specific":
            return self._solve_q_arm_specific()

        A_mat = self.A.reshape(-1, 1)
        N_mat = np.hstack([self.ones, self.W, A_mat, self.X])

        idx_A = 1 + self.W.shape[1]
        C = np.zeros(N_mat.shape[1])
        C[idx_A] = 1.0

        dim_t = 1 + self.Z.shape[1] + 1 + self.X.shape[1]

        # 定义方程组
        def equations(t):

            q_val = self.q_func(t, self.Z, self.A, self.X)

            sign = (2 * self.A - 1)

            term = (sign * q_val)[:, None] * N_mat

            sample_mean = np.mean(term, axis=0)

            return sample_mean - C

        t0 = np.zeros(dim_t)

        sol = optimize.root(equations, t0, method='lm')
        if not sol.success:
            print("Warning: q solver did not converge:", sol.message)

        residual_norm = float(np.linalg.norm(sol.fun))
        return sol.x, bool(sol.success), str(sol.message), residual_norm

    def estimate_ate(self):

        b_hat, _used_lstsq = self.solve_h()
        t_hat, _converged, _message, _residual_norm = self.solve_q()

        n = self.n

        h1 = self.h_func(b_hat, self.W, np.ones(n), self.X)
        h0 = self.h_func(b_hat, self.W, np.zeros(n), self.X)
        psi_por = np.mean(h1 - h0)
        
        # --- PIPW (Proximal IPW) ---
        # psi = E[ (-1)^(1-A) * q(Z,A,X) * Y ]
        q_val = self.q_func(t_hat, self.Z, self.A, self.X)
        sign = (2 * self.A - 1)
        psi_pipw = np.mean(sign * q_val * self.Y)
        
        # --- PDR (Proximal Doubly Robust) ---
        # psi = E[ (-1)^(1-A)*q*(Y - h) + h(1) - h(0) ]
        h_obs = self.h_func(b_hat, self.W, self.A, self.X)
        
        term1 = sign * q_val * (self.Y - h_obs)
        term2 = h1 - h0
        psi_pdr = np.mean(term1 + term2)
        
        return {
            "POR": psi_por,
            "PIPW": psi_pipw,
            "PDR": psi_pdr,
            "h_params": b_hat,
            "q_params": t_hat
        }

    # ------------------------------------------------------------------
    # Arm-specific q (eq. 14 with all Z-A and X-A interactions)
    # ------------------------------------------------------------------

    def _arm_features(self, Z, X) -> np.ndarray:
        return np.hstack([np.ones((Z.shape[0], 1)), Z, X])

    def _q_arm_specific(self, t, Z, A, X) -> Tuple[np.ndarray, np.ndarray]:
        """
        q at the given A, and its gradient with respect to t = (theta_0, theta_1)
        (one row per sample, zero in the block of the other arm).
        """
        F = self._arm_features(Z, X)
        p = F.shape[1]
        theta0, theta1 = t[:p], t[p:]
        a1 = (np.asarray(A).reshape(-1) > 0.5)
        index = np.where(a1, F @ theta1, F @ theta0)
        e = np.exp(index)
        grad = np.zeros((F.shape[0], 2 * p))
        grad[~a1, :p] = e[~a1, None] * F[~a1]
        grad[a1, p:] = e[a1, None] * F[a1]
        return 1.0 + e, grad

    def _solve_q_arm_specific(self):
        """
        For each arm a, solve E_n{1(A=a) q_a (1,W,X)' - (1,W,X)'} = 0 by
        Levenberg-Marquardt with the analytic Jacobian. Z and X are
        standardized inside the solver and the coefficients mapped back,
        which leaves the fitted q unchanged (affine reparametrization of an
        exactly identified system) but makes the problem well scaled.
        Starts from q_a = 1 / P_n(A=a).
        """
        F = self._arm_features(self.Z, self.X)
        G = np.hstack([self.ones, self.W, self.X])
        mu = F[:, 1:].mean(axis=0)
        sd = F[:, 1:].std(axis=0)
        sd[sd == 0] = 1.0
        F_s = np.hstack([self.ones, (F[:, 1:] - mu) / sd])
        g_mu = G[:, 1:].mean(axis=0)
        g_sd = G[:, 1:].std(axis=0)
        g_sd[g_sd == 0] = 1.0
        G_s = np.hstack([self.ones, (G[:, 1:] - g_mu) / g_sd])
        target = G_s.mean(axis=0)

        thetas, successes, messages, residuals = [], [], [], []
        for a in (0, 1):
            mask = ((self.A > 0.5) == bool(a)).astype(float)
            p_a = mask.mean()

            def equations(th, mask=mask):
                q = 1.0 + np.exp(F_s @ th)
                return (mask * q) @ G_s / self.n - target

            def jacobian(th, mask=mask):
                e = np.exp(F_s @ th)
                return (G_s * (mask * e)[:, None]).T @ F_s / self.n

            start = np.zeros(F_s.shape[1])
            start[0] = np.log(1.0 / p_a - 1.0)
            sol = optimize.root(equations, start, jac=jacobian, method='lm')
            th_s = sol.x
            slopes = th_s[1:] / sd
            intercept = th_s[0] - float(np.sum(slopes * mu))
            thetas.append(np.concatenate([[intercept], slopes]))
            successes.append(bool(sol.success))
            messages.append(str(sol.message))
            residuals.append(sol.fun)

        t_opt = np.concatenate(thetas)
        # Residual on the original (unstandardized) moment equations.
        residual_norm = float(np.linalg.norm(self.q_moment_rows(t_opt).mean(axis=0)))
        converged = all(successes)
        if not converged:
            print("Warning: q solver did not converge:", messages)
        return t_opt, converged, "; ".join(messages), residual_norm

    # ------------------------------------------------------------------
    # Estimating functions and the stacked sandwich variance
    # ------------------------------------------------------------------

    def _q_and_grad(self, t, Z, A, X) -> Tuple[np.ndarray, np.ndarray]:
        """q(Z,A,X;t) at the given A and dq/dt (n x dim_t)."""
        if self.q_form == "arm_specific":
            return self._q_arm_specific(t, Z, A, X)
        A_mat = A.reshape(-1, 1)
        F = np.hstack([np.ones((A_mat.shape[0], 1)), Z, A_mat, X])
        sign = (2 * A - 1)
        e = np.exp(sign * (F @ t))
        return 1.0 + e, (sign * e)[:, None] * F

    def h_moment_rows(self, b) -> np.ndarray:
        M = np.hstack([self.ones, self.Z, self.A.reshape(-1, 1), self.X])
        V = np.hstack([self.ones, self.A.reshape(-1, 1), self.W, self.X])
        return (self.Y - V @ b)[:, None] * M

    def q_moment_rows(self, t) -> np.ndarray:
        """Per-sample q estimating functions; their mean is zero at t_hat."""
        q, _ = self._q_and_grad(t, self.Z, self.A, self.X)
        if self.q_form == "pooled":
            N = np.hstack([self.ones, self.W, self.A.reshape(-1, 1), self.X])
            C = np.zeros(N.shape[1])
            C[1 + self.W.shape[1]] = 1.0
            return ((2 * self.A - 1) * q)[:, None] * N - C
        G = np.hstack([self.ones, self.W, self.X])
        a1 = (self.A > 0.5)
        rows0 = (~a1 * q)[:, None] * G - G
        rows1 = (a1 * q)[:, None] * G - G
        return np.hstack([rows0, rows1])

    def _q_moment_jacobian(self, t) -> np.ndarray:
        """d mean(q_moment_rows) / dt."""
        _, dq = self._q_and_grad(t, self.Z, self.A, self.X)
        if self.q_form == "pooled":
            N = np.hstack([self.ones, self.W, self.A.reshape(-1, 1), self.X])
            return (N * (2 * self.A - 1)[:, None]).T @ dq / self.n
        G = np.hstack([self.ones, self.W, self.X])
        a1 = (self.A > 0.5)
        return np.vstack([(G * (~a1)[:, None]).T @ dq, (G * a1[:, None]).T @ dq]) / self.n

    def estimate_ate_with_se(self) -> Dict[str, Any]:
        """
        Full-sample POR, PIPW and PDR with standard errors from the stacked
        M-estimation sandwich (h, q and the three effect equations solved
        jointly), as in Cui et al. (2024), Section 5.

        Returns estimates, standard errors, 95% Wald intervals, the fitted
        parameters and solver diagnostics.
        """
        b_hat, used_lstsq = self.solve_h()
        t_hat, q_converged, q_message, q_residual_norm = self.solve_q()
        n = self.n
        s = 2 * self.A - 1

        V = np.hstack([self.ones, self.A.reshape(-1, 1), self.W, self.X])
        V1 = np.hstack([self.ones, np.ones((n, 1)), self.W, self.X])
        V0 = np.hstack([self.ones, np.zeros((n, 1)), self.W, self.X])
        h_obs = V @ b_hat
        h_diff = (V1 - V0) @ b_hat
        q, dq = self._q_and_grad(t_hat, self.Z, self.A, self.X)

        psi = np.array([
            np.mean(h_diff),
            np.mean(s * q * self.Y),
            np.mean(s * q * (self.Y - h_obs) + h_diff),
        ])
        psi_rows = np.column_stack([
            h_diff - psi[0],
            s * q * self.Y - psi[1],
            s * q * (self.Y - h_obs) + h_diff - psi[2],
        ])
        rows = np.hstack([self.h_moment_rows(b_hat), self.q_moment_rows(t_hat), psi_rows])

        db, dt = self.dim_b, self.dim_t
        p = db + dt + 3
        J = np.zeros((p, p))
        M = np.hstack([self.ones, self.Z, self.A.reshape(-1, 1), self.X])
        J[:db, :db] = -(M.T @ V) / n
        J[db:db + dt, db:db + dt] = self._q_moment_jacobian(t_hat)
        r = db + dt
        J[r, :db] = (V1 - V0).mean(axis=0)
        J[r + 1, db:db + dt] = (s * self.Y) @ dq / n
        J[r + 2, :db] = (-(s * q)[:, None] * V + (V1 - V0)).mean(axis=0)
        J[r + 2, db:db + dt] = (s * (self.Y - h_obs)) @ dq / n
        J[r:, r:] = -np.eye(3)

        S = rows.T @ rows / n
        J_inv = np.linalg.inv(J)
        cov = J_inv @ S @ J_inv.T / n
        se = np.sqrt(np.diag(cov)[r:])
        names = ["POR", "PIPW", "PDR"]
        return {
            "estimates": dict(zip(names, psi)),
            "se": dict(zip(names, se)),
            "ci95": {k: (v - 1.959963984540054 * e, v + 1.959963984540054 * e)
                     for k, v, e in zip(names, psi, se)},
            "h_params": b_hat,
            "q_params": t_hat,
            "used_lstsq": bool(used_lstsq),
            "q_converged": bool(q_converged),
            "q_message": q_message,
            "q_residual_norm": q_residual_norm,
            "jacobian_condition": float(np.linalg.cond(J)),
        }

def LINEAR_experiment(data_configs: Dict[str, Any], data_seed: int = 0):
    output_path = data_configs.get('output_path', 'predicts/linear')
    n_samples = data_configs.get('n_samples', 2000)
    n_trials = data_configs.get('n_trials', 1)
    scenario = data_configs.get('scenario', 1)
    POR = []
    PIPW = []
    PDR = []
    for trial_idx in range(n_trials):
        trial_seeds = make_seed_bundle(data_seed, trial_idx)
        df = generate_data(n_samples=n_samples, scenario=scenario, seed=trial_seeds.data_seed)
        model = ProximalLinearModel(df)
        est = model.estimate_ate()
        POR.append(est['POR'])
        PIPW.append(est['PIPW'])
        PDR.append(est['PDR'])
    results = pd.DataFrame({
        "POR": POR,
        "PIPW": PIPW,
        "PDR": PDR
    })
    
    print(f"\n--- 线性模型 ATE 估计结果 (平均 over {n_trials} trials) ---")
    print(f"POR = {results['POR'].mean():.4f}")
    print(f"PIPW = {results['PIPW'].mean():.4f}")
    print(f"PDR = {results['PDR'].mean():.4f}")
    print("----------------------------------")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_file = output_path / f"linear_results_{timestamp}.csv"
    results.to_csv(out_file, index=False)


    
