import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import entropy
import warnings

import os, sys

from causalml.inference.meta import LRSRegressor
from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor
from causalml.inference.tf import DragonNet
from causalml.match import NearestNeighborMatch, MatchOptimizer, create_table_one
from causalml.propensity import ElasticNetPropensityModel
from causalml.metrics import *

warnings.filterwarnings('ignore')
import tensorflow as tf

import random
def set_seed(seed=42):
    """Set seed for reproducibility across Python, NumPy, and TensorFlow."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Enforce deterministic behavior in TF (as much as possible)

    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    tf.random.set_seed(seed)  # TensorFlow

set_seed(seed=42)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first visible GPU (which is CUDA:3 due to CUDA_VISIBLE_DEVICES)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
plt.style.use('fivethirtyeight')
sns.set_palette('Paired')
plt.rcParams['figure.figsize'] = (12,8)

df = pd.read_csv(f'docs/examples/data/ihdp_npci_3.csv', header=None)
cols =  ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f'x{i}' for i in range(1,26)]
df.columns = cols

pd.Series(df['treatment']).value_counts(normalize=True)

X = df.loc[:,'x1':]
treatment = df['treatment']
y = df['y_factual']
tau = df.apply(lambda d: d['y_factual'] - d['y_cfactual'] if d['treatment']==1 
               else d['y_cfactual'] - d['y_factual'], 
               axis=1)

# Split into train and test for IHDP data
X_train, X_test, treatment_train, treatment_test, y_train, y_test, tau_train, tau_test = train_test_split(
    X, treatment, y, tau, test_size=0.1, random_state=1, shuffle=True
) # using 0.1 of the data to test

print("Fitting ElasticNetPropensityModel...")
p_model = ElasticNetPropensityModel()
p = p_model.fit_predict(X, treatment)
print("Propensity scores estimated.")

print("Training DragonNet on IHDP training set...")
dragon = DragonNet(neurons_per_layer=200, val_split=0.3, targeted_reg=True) # using 0.9*0.7=0.63 of the data to train, 0.27 for validation
dragon.fit(X_train, treatment_train, y_train)
print("DragonNet training complete.")

print("Predicting on IHDP test set...")
dragon_ite_test = dragon.predict_tau(X_test).flatten()
dragon_ate_test = dragon_ite_test.mean()

print("Preparing test prediction DataFrame...")
df_preds_test = pd.DataFrame([dragon_ite_test.ravel(),
                              tau_test.ravel(),
                              treatment_test.ravel(),
                              y_test.ravel()],
                             index=['dragonnet','tau','w','y']).T

print("Calculating cumulative gain on test set...")
df_cumgain_test = get_cumgain(df_preds_test)

print("Compiling test results...")
df_result_test = pd.DataFrame([dragon_ate_test, tau_test.mean()],
                              index=['dragonnet','actual'], columns=['ATE'])
df_result_test['MAE'] = [mean_absolute_error(tau_test, dragon_ite_test.ravel())] + [None]
df_result_test['AUUC'] = auuc_score(df_preds_test)

# Ensure results_tmp directory exists
os.makedirs('results_tmp', exist_ok=True)

# Save test results to CSV
df_result_test.to_csv("results_tmp/results_summary_test.csv")
print("Test results saved to results_tmp/results_summary_test.csv")

# Print test results to console
print("IHDP Test Results Summary:")
print(df_result_test)

print("Plotting gain curve (test set)...")
plot_gain(df_preds_test)
plt.savefig("results_tmp/gain_curve_test.png")
print("Gain curve saved to results_tmp/gain_curve_test.png")
print("IHDP experiment complete.")

# --- Synthetic data part with printing and saving ---
print("\n=== Synthetic Data Experiment ===")
print("Generating synthetic data using simulate_nuisance_and_easy_treatment...")

from scipy.special import expit, logit
def simulate_nuisance_and_easy_treatment(n=1000, p=5, sigma=1.0, adj=0.0, seed=None):
    """Synthetic data with a difficult nuisance components and an easy treatment effect
        From Setup A in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
        seed (int, optional): random seed for reproducibility
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.uniform(size=n * p).reshape((n, -1))
    b = (
        np.sin(np.pi * X[:, 0] * X[:, 1])
        + 2 * (X[:, 2] - 0.5) ** 2
        + X[:, 3]
        + 0.5 * X[:, 4]
    )
    eta = 0.1
    e = np.maximum(
        np.repeat(eta, n),
        np.minimum(np.sin(np.pi * X[:, 0] * X[:, 1]), np.repeat(1 - eta, n)),
    )
    e = expit(logit(e) - adj)
    tau = (X[:, 0] + X[:, 1]) / 2

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e

y, X, w, tau, b, e = simulate_nuisance_and_easy_treatment(n=1000, seed=42) # add random seed for reproducibility
print("Synthetic data generated.")
print(f"Shapes: X={X.shape}, y={y.shape}, w={w.shape}, tau={tau.shape}")

X_train, X_test, y_train, y_test, w_train, w_test, tau_train, tau_test, b_train, b_test, e_train, e_test = \
    train_test_split(X, y, w, tau, b, e, test_size=0.1, random_state=1, shuffle=True) # using 0.1 of the data to test
print("Split synthetic data into train and test sets.")
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Train DragonNet
print("Training DragonNet on synthetic training set...")
dragonnet = DragonNet(neurons_per_layer=200, val_split=0.3, targeted_reg=True) # using 0.9*0.7=0.63 of the data to train, 0.27 for validation
dragonnet.fit(X_train, treatment=w_train, y=y_train)
print("DragonNet training complete.")

tau_pred_test = dragonnet.predict_tau(X=X_test).flatten()

# Summarize results for test set only
ate_test = tau_pred_test.mean()
ate_actual_test = tau_test.mean()
mse_test = mse(tau_pred_test, tau_test)
abs_perc_error_ate_test = np.abs((ate_test / ate_actual_test) - 1)
# KL divergence for test
stacked_values_test = np.hstack((tau_pred_test, tau_test))
stacked_low_test = np.percentile(stacked_values_test, 0.1)
stacked_high_test = np.percentile(stacked_values_test, 99.9)
bins_test = np.linspace(stacked_low_test, stacked_high_test, 100)
distr_test = np.histogram(tau_pred_test, bins=bins_test)[0]
distr_test = np.clip(distr_test/distr_test.sum(), 0.001, 0.999)
true_distr_test = np.histogram(tau_test, bins=bins_test)[0]
true_distr_test = np.clip(true_distr_test/true_distr_test.sum(), 0.001, 0.999)
kl_test = entropy(distr_test, true_distr_test)

df_preds_test = pd.DataFrame([tau_pred_test.ravel(),
                              tau_test.ravel(),
                              w_test.ravel(),
                              y_test.ravel()],
                             index=['DragonNet','tau','w','y']).T
auuc_test = auuc_score(df_preds_test).iloc[0]

synthetic_summary_test = pd.DataFrame({
    'ATE': [ate_test, ate_actual_test],
    'MSE': [mse_test, 0],  # Actual has MSE=0 since it's ground truth
    'Abs % Error of ATE': [abs_perc_error_ate_test, 0],  # Actual has 0% error
    'KL Divergence': [kl_test, 0],  # Actual has KL=0 since it matches itself
    'AUUC': [auuc_test, np.nan]  # AUUC is NaN for actuals since it's ground truth
}, index=['DragonNet', 'Actuals'])

print("\nSynthetic Test Set Results:")
print(synthetic_summary_test)
synthetic_summary_test.to_csv("results_tmp/synthetic_test_results.csv")
print("Synthetic test results saved to results_tmp/synthetic_test_results.csv")

# Save detailed predictions and actuals for test set
test_predictions_df = pd.DataFrame({
    'predicted_tau': tau_pred_test,
    'actual_tau': tau_test,
    'treatment': w_test,
    'outcome': y_test
})
test_predictions_df.to_csv("results_tmp/synthetic_test_predictions.csv", index=False)
print("Detailed test predictions saved to results_tmp/synthetic_test_predictions.csv")

print("Plotting synthetic gain curve for test set...")
plot_gain(df_preds_test)
plt.savefig("results_tmp/synthetic_gain_curve_test.png")
print("Synthetic gain curve for test set saved to results_tmp/synthetic_gain_curve_test.png")
print("Synthetic experiment complete.")

print("\nFormatted IHDP Test Set Results:")
print(f"IHDP ATE: {df_result_test.loc['dragonnet', 'ATE']:.6f}")
print(f"IHDP MAE: {df_result_test.loc['dragonnet', 'MAE']:.6f}")
print(f"IHDP AUUC: {df_result_test.loc['dragonnet', 'AUUC']:.6f}")

print("\nFormatted Synthetic Test Set Results:")
print(f"Synthetic ATE: {synthetic_summary_test.loc['DragonNet', 'ATE']:.6f}")
print(f"Synthetic MSE: {synthetic_summary_test.loc['DragonNet', 'MSE']:.6f}")
print(f"Synthetic Abs % Error of ATE: {synthetic_summary_test.loc['DragonNet', 'Abs % Error of ATE']:.6f}")
print(f"Synthetic KL Divergence: {synthetic_summary_test.loc['DragonNet', 'KL Divergence']:.6f}")
print(f"Synthetic AUUC: {synthetic_summary_test.loc['DragonNet', 'AUUC']:.6f}")
