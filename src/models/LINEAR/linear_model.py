import numpy as np
import pandas as pd
from scipy import optimize
from pathlib import Path

import time
from src.data.ate.sgd_pv import generate_data
from src.utils import make_seed_bundle
from typing import Dict, Any

class ProximalLinearModel:
    def __init__(self, df):
        """
        初始化数据
        假设 df 包含列: A, Y, Z_1...Z_dz, W_1...W_dw, X_1...X_dx
        在 SGD 数据集中:
        A: 1维
        Y: 1维
        Z: 1维 
        W: 1维
        X: 2维 (X1, X2)
        """
        self.n = len(df)

        self.A = df['A'].values
        self.Y = df['Y'].values
 
        if 'Z' in df.columns:
            self.Z = df[['Z']].values # shape (n, 1)
        else:
            z_cols = [c for c in df.columns if c.startswith('Z')]
            self.Z = df[z_cols].values

        if 'W' in df.columns:
            self.W = df[['W']].values
        else:
            w_cols = [c for c in df.columns if c.startswith('W')]
            self.W = df[w_cols].values
            
        if 'X1' in df.columns:
            self.X = df[['X1', 'X2']].values
        else:
            self.X = np.zeros((self.n, 0))

        self.ones = np.ones((self.n, 1))

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
        """
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
        solve was singular and the lstsq fallback was used -- reported as a
        diagnostic only (plan/p0_e1_e2's Phase "Oracle/参数Proximal" round:
        "记录是否使用了 lstsq，不要因为条件数较大而自动判失败").
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
        """

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


    
