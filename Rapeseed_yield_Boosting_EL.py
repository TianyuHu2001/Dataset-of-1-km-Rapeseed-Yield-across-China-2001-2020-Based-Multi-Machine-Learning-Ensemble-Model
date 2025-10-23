#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os

import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
import lightgbm as LGBM
import xgboost as XGB
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin, clone  # 新增
from matplotlib.ticker import FuncFormatter

# Constructing the Boosting
class ResidualBoostingEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, learning_rate=0.5, meta_learning_rate=0.5,
                 meta_xgb_params=None, random_state=1):
        self.estimators = estimators
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        self.meta_xgb_params = meta_xgb_params or {
            "n_estimators": 1350,
            "max_depth": 8,
            "learning_rate": 0.04,
            "subsample": 0.77,
            "colsample_bytree": 0.94,
            "reg_lambda": 0.002,
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": 0
        }
        self.random_state = random_state

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        self.n_features_in_ = X.shape[1]
        if isinstance(self.estimators, list) and len(self.estimators) > 0 and isinstance(self.estimators[0], tuple):
            name_model_pairs = self.estimators
        elif isinstance(self.estimators, dict):
            name_model_pairs = list(self.estimators.items())
        else:
            name_model_pairs = [(f"m{i}", m) for i, m in enumerate(self.estimators)]
        self.named_estimators_ = {}
        self._fitted_sequence_ = []
        stage_preds = []

        # First:fitting Y
        name1, model1 = name_model_pairs[0]
        m1 = clone(model1)
        m1.fit(X, y)
        y_hat = m1.predict(X)
        stage_preds.append(y_hat.copy())
        self.named_estimators_[name1] = m1
        self._fitted_sequence_.append((name1, m1))

        # Next：learning model bias
        for name, base_m in name_model_pairs[1:]:
            m = clone(base_m)
            residual = y - y_hat
            m.fit(X, residual)
            y_hat = y_hat + self.learning_rate * m.predict(X)
            stage_preds.append(y_hat.copy())
            self.named_estimators_[name] = m
            self._fitted_sequence_.append((name, m))
        f1 = self._fitted_sequence_[0][1].predict(X)
        f2 = self._fitted_sequence_[1][1].predict(X)
        f3 = self._fitted_sequence_[2][1].predict(X)
        Z = np.column_stack([f1, f2, f3])
        y_hat_curr = f1 + self.learning_rate * f2 + self.learning_rate * f3
        final_residual = y - y_hat_curr
        self.meta_model_ = XGB.XGBRegressor(**self.meta_xgb_params)
        self.meta_model_.fit(Z, final_residual)
        self._has_meta_ = True
        return self
    def predict(self, X):
        # First model fitting
        y1 = self._fitted_sequence_[0][1].predict(X)
        # Second model fitting including the first model bias
        y2_res = self._fitted_sequence_[1][1].predict(X)
        # Third model fitting including the Second model bias
        y3_res = self._fitted_sequence_[2][1].predict(X)

        # Final model train
        y_hat = y1 + self.learning_rate * y2_res + self.learning_rate * y3_res

        # Base-models fitting result and bias to train meta_model
        if getattr(self, "_has_meta_", False):
            Z = np.column_stack([y1, y2_res, y3_res])
            corr = self.meta_model_.predict(Z)
            y_hat = y_hat + self.meta_learning_rate * corr
        return y_hat

# 设置统一绘图风格
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 23
plt.rcParams['xtick.labelsize'] = 23
plt.rcParams['ytick.labelsize'] = 24

# Unified the plots style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 23
plt.rcParams['xtick.labelsize'] = 23
plt.rcParams['ytick.labelsize'] = 24

# Read the data
print("Start reading data...")
data_train = pd.read_excel("Sample.xlsx")
data_train.dropna(inplace=True)
print("Finish reading data。")

# Characteristic data splitting
print("Start splitting data...")
X = data_train.drop(columns=['Yield', 'qhdm', 'name'])
y = data_train['Yield']
print("Finish splitting data")

# Five-fold
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.2, random_state=4)

# Base-models training
forest = RandomForestRegressor(n_estimators=350, max_depth=25, max_features=0.8, min_samples_split=2,
                               min_samples_leaf=1, bootstrap=True, random_state=1, n_jobs=-1)
forest.fit(X_train, Y_train)

LGBM = LGBM.LGBMRegressor(n_estimators=1000, max_depth=25, learning_rate=0.05, subsample=0.79, colsample_bytree=0.73,
                          num_leaves=144, reg_lambda=0.04, random_state=1, n_jobs=-1, verbosity=-1)
LGBM.fit(X_train, Y_train)

XGB_model = XGB.XGBRegressor(n_estimators=1350, max_depth=8, learning_rate=0.04, subsample=0.77, colsample_bytree=0.94,
                             min_child_weight=2, reg_lambda=0.002)
XGB_model.fit(X_train, Y_train)

# Constructing model sequence in the Boosting
print("Boosting-EL（XGB → LGBM → RF + final XGB as meta-model）...")
estimators = [('XGB', XGB_model), ('LGBM', LGBM), ('RF', forest)]
boosting_regressor = ResidualBoostingEnsemble(
    estimators=estimators, learning_rate=0.5, meta_learning_rate=0.5,
    meta_xgb_params={"n_estimators": 1350, "max_depth": 8, "learning_rate": 0.04,
                     "subsample": 0.77, "colsample_bytree": 0.94, "reg_lambda": 0.002,
                     "random_state": 1, "n_jobs": -1, "verbosity": 0}
)
boosting_regressor.fit(X_train, Y_train)

# excluding the extreme values of sample
def evaluate_model(name, y_true, y_pred, threshold_percentile=95):
    abs_error = np.abs(y_pred - y_true)
    threshold = np.percentile(abs_error, threshold_percentile)
    mask = abs_error <= threshold
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    mae = mean_absolute_error(y_true_filtered, y_pred_filtered)
    mse = mean_squared_error(y_true_filtered, y_pred_filtered)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_filtered, y_pred_filtered)
    print(f"{name} internal-validation（only {threshold_percentile}% sample）:")
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

# Performance assessment of base-models
y_pred_RF = forest.predict(X_validation)
metrics_RF = evaluate_model("RF", Y_validation, y_pred_RF)

y_pred_LGBM = LGBM.predict(X_validation)
metrics_LGBM = evaluate_model("LightGBM", Y_validation, y_pred_LGBM)

y_pred_XGB = XGB_model.predict(X_validation)
metrics_XGB = evaluate_model("XGBoost", Y_validation, y_pred_XGB)

# Boosting
y_pred_boosting = boosting_regressor.predict(X_validation)
metrics_boosting = evaluate_model("Boosting-EL", Y_validation, y_pred_boosting)

# Print the model performance
def print_metrics(name, metrics):
    print(name)
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R²: {metrics['R2']:.4f}\n")

print_metrics("XGBoost", metrics_XGB)
print_metrics("LightGBM", metrics_LGBM)
print_metrics("RF", metrics_RF)
print_metrics("Boosting_EL", metrics_boosting)

# 特征重要性（函数原样保留；打印每个“阶段模型”的重要性；终层 XGB 是 meta，不在该字典里）
def print_boosting_model_importances(boosting_model, X_train):
    for name, model in boosting_model.named_estimators_.items():
        if hasattr(model, 'Predictor_importances_'):
            print(f"\n{name} Predictor importances：")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 8))
            plt.title(f"{name} Predictor Importances", fontsize=24, weight='bold')
            plt.bar(range(X_train.shape[1]), importances[indices], align='center')
            plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=45, fontsize=13, weight='bold')
            plt.xlabel("Predictors", fontsize=20, weight='bold')
            plt.ylabel("Importance", fontsize=20, weight='bold')
            plt.tight_layout()
            plt.show()
print_boosting_model_importances(boosting_regressor, X_train)

# Partial dependence plots of the predictors in the boosting
def plot_pdp(X_train, model, features):
    num_columns = 3
    num_rows = (len(features) + num_columns - 1) // num_columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 4))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        PartialDependenceDisplay.from_estimator(model, X_train, [feature], percentiles=(0, 1), ax=axes[i])
        axes[i].set_title(f"PDP: {feature}", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()
plot_pdp(X_train, boosting_regressor, X_train.columns)

from scipy.stats import gaussian_kde

# Scatter plots
def plot_density_scatter(y_true, y_pred, model_name="Model", threshold_percentile=95):
    abs_error = np.abs(y_pred - y_true)
    threshold = np.percentile(abs_error, threshold_percentile)
    mask = abs_error <= threshold
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    r2 = r2_score(y_true_filtered, y_pred_filtered)
    rmse = np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))

    # Sample density calculate
    xy = np.vstack([y_true_filtered, y_pred_filtered])
    z = gaussian_kde(xy)(xy)

    # Drawing plot
    plt.figure(figsize=(8, 7))
    ax = plt.gca()
    scatter = ax.scatter(y_true_filtered, y_pred_filtered, c=z, s=40,
                         cmap='viridis', edgecolors='none')
    ax.plot([min(y_true_filtered), max(y_true_filtered)],
            [min(y_true_filtered), max(y_true_filtered)],
            color='black', linestyle='--', label='45° Line')
    fit_line = np.poly1d(np.polyfit(y_true_filtered, y_pred_filtered, 1))
    ax.plot(y_true_filtered, fit_line(y_true_filtered), color='red', label='Fitted Line')
    ax.set_title(f'{model_name}', fontsize=22, weight='bold')
    ax.set_xlabel('Statistical Yield(×10³ kg/ha)', fontsize=24, weight='bold')
    ax.set_ylabel('Predicted Yield(×10³ kg/ha)', fontsize=24, weight='bold')
    def thousands_formatter(x, pos):
        val = x / 1000
        if abs(val) < 5e-13:
            val = 0.0
        return f'{val:.1f}'
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.colorbar(scatter, ax=ax, label='Density')
    leg = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True)
    fig = ax.figure
    fig.canvas.draw()
    bbox_ax = leg.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
    pad = 0.02
    x_text = 0.02
    y_text = max(bbox_ax.y0 - pad, 0.02)
    ax.text(x_text, y_text,
            f'R² = {r2:.2f}\nRMSE = {rmse:.2f}',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=24, bbox=dict(facecolor='white', alpha=0.6))
    plt.tight_layout()
    plt.show()
plot_density_scatter(Y_validation, y_pred_boosting, model_name="Boosting")

base_path = r'F:\油菜'
output_path = r'/SD-yield/predict-yield-250914'
years = [f'{y:02d}' for y in range(1, 21)]  # '01' 到 '20'

for y in years:
    print(f"processing year：20{y}")
    try:
        # reading raster data
        with rasterio.open(os.path.join(base_path, f'rape240621\\nian\\tongyi\\tongyinian{y}S.tif')) as src:  # Year
            Year = src.read(1)
            meta = src.meta
        with rasterio.open(os.path.join(base_path, 'rape240621\\dem\\maskdem_2km2S.tif')) as src:  # DEM
            DEM = src.read(1)
        with rasterio.open(os.path.join(base_path, f'rape240621\\LAI\\2000\\LAIgsmax\\mask\\clean\\lai20{y}new.tif')) as src:  # LAI
            LAI = src.read(1)
        with rasterio.open(os.path.join(base_path, f'rape240621\\tr\\tongyi\\tongyiTNreS.tif')) as src:  # TN
            TN = src.read(1)
        with rasterio.open(os.path.join(base_path, f'rape240621\\tr\\tongyi\\tongyiTPreS.tif')) as src:  # TP
            TP = src.read(1)
        with rasterio.open(os.path.join(base_path, f'rape240621\\sif\\sif\\sif_bianliang\\sif_ty\\sif{y}newS.tif')) as src:  # SIF
            SIF = src.read(1)
        with rasterio.open(os.path.join(base_path, f'rape240621\\pregssum\\2001\\mask\\pre20{y}new.tif')) as src:  # Pre
            Pre = src.read(1)
        with rasterio.open(os.path.join(base_path, f'rape240621\\tmaxgsmean\\mask\\tmax20{y}new.tif')) as src:  # Tmax
            Tmax = src.read(1)
        with rasterio.open(os.path.join(base_path, f'rape240621\\tmingsmean\\mask\\tmin20{y}new.tif')) as src:  # Tmin
            Tmin = src.read(1)

        # Stack the predictor
        X = np.stack((Year, DEM, LAI, TN, TP, SIF, Pre, Tmax, Tmin), axis=-1)
        rows, cols, bands = X.shape
        X_2d = X.reshape(rows * cols, bands)

        # Predict by using boosting-EL
        y_pred = boosting_regressor.predict(X_2d)
        out_file = os.path.join(output_path, f'predict{y}_boosting.tif')
        with rasterio.open(out_file, 'w', **meta) as dst:
            dst.write(y_pred.reshape(rows, cols), 1)
        print(f"year 20{y} finished！")
    except Exception as e:
        print(f"year 20{y} processing error：{e}")