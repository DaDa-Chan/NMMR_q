import numpy as np
import pandas as pd
from pathlib import Path

TRUE_ATE = 2

def generate_data(n_samples=2000):
    # -- X --
    gamma_x = np.array([0.25, 0.25])
    cov_x = np.array([[0.25**2, 0], [0, 0.25**2]])
    X = np.random.multivariate_normal(gamma_x, cov_x, n_samples)

    # -- A --
    p_a = 1 / (1 + np.exp(-(0.125 * X[:, 0] + 0.125 * X[:, 1])))
    A = np.random.binomial(1, p_a, n_samples)

    mean_zwu = np.zeros((n_samples, 3))
    alpha_x = mu_x = kappa_x = np.array([0.25, 0.25])
    mean_zwu[:, 0] = 0.25 + 0.25 * A + X @ alpha_x
    mean_zwu[:, 1] = 0.25 + 0.125 * A + X @ mu_x
    mean_zwu[:, 2] = 0.25 + 0.25 * A + X @ kappa_x
    cov_zwu = np.array([[1, 0.25, 0.5], [0.25, 1, 0.5], [0.5, 0.5, 1]])

    # -- Z, W, U --
    results_zwu = np.random.multivariate_normal([0, 0, 0], cov_zwu, n_samples) + mean_zwu
    Z = results_zwu[:, 0]
    W = results_zwu[:, 1]
    U = results_zwu[:, 2]

    bx = np.array([0.25, 0.25])

    # -- E(W|U,X) --
    E_W_given_U_X = 0.25 + X @ mu_x + (cov_zwu[1, 2] / cov_zwu[2, 2]) * (U - 0.25 - X @ kappa_x)

    # -- E(Y|W,U,A,Z,X) --
    E_Y_given_W_U_A_Z_X = 2 + 2 * A + X @ bx + (4 - 2) * E_W_given_U_X + 2 * W

    # -- Y --
    Y = E_Y_given_W_U_A_Z_X + np.random.normal(0, 0.25, n_samples)
    
    return pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'A': A, 'Z': Z, 'W': W, 'U': U, 'Y': Y})

n_train = 10000
train_df = generate_data(n_samples=n_train)
n_cf = 10000
cf_df = generate_data(n_samples=n_cf)

output_dir = Path('/Users/chen/Study/CI/NMMR_q/data/SGD')
output_dir.mkdir(parents=True, exist_ok=True)
sgd_train = output_dir / 'sgd_train.csv'
sgd_cf = output_dir / 'sgd_cf.csv'

train_df.to_csv(sgd_train, index=False)
cf_df.to_csv(sgd_cf, index=False)








