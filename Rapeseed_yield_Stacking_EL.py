#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from seaborn.external.kde import gaussian_kde
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
import lightgbm as LGBM
import xgboost as XGB
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib.ticker import FuncFormatter

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

# —— Stacking —— #
stacking_model = StackingRegressor(
    estimators=[("RF", forest), ("LGBM", LGBM), ("XGB", XGB_model)],
    final_estimator=XGB.XGBRegressor(
        n_estimators=1350, max_depth=8, learning_rate=0.04,
        subsample=0.77, colsample_bytree=0.94,
        min_child_weight=2, reg_lambda=0.002
    ),
    passthrough=False, cv="prefit"
)
stacking_model.fit(X_train, Y_train)
y_pred_stacking = stacking_model.predict(X_validation)
metrics_stacking = evaluate_model("Stacking", Y_validation, y_pred_stacking)

# Print the model performance
def print_metrics(name, metrics):
    print(name)
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R²: {metrics['R2']:.4f}\n")

print_metrics("RF", metrics_RF)
print_metrics("LightGBM", metrics_LGBM)
print_metrics("XGBoost", metrics_XGB)
print_metrics("Stacking", metrics_stacking)
# Predictors importance of based-models
def print_base_model_importances(models, model_names, X_train):
    for name, model in zip(model_names, models):
        if hasattr(model, 'Predictor_importances_'):
            print(f"\n{name} Predictor importances:")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 8))
            plt.title(f"{name} Predictor Importances", fontsize=22, weight='bold')
            plt.bar(range(X_train.shape[1]), importances[indices], align='center')
            plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=45, fontsize=20, weight='bold')
            plt.xlabel("Predictors", fontsize=24, weight='bold')
            plt.ylabel("Importance", fontsize=24, weight='bold')
            plt.tight_layout()
            plt.show()

print_base_model_importances(
    [forest, LGBM, XGB_model],
    ["RF", "LGBM", "XGB"],
    X_train
)

# Partial dependence plots of the predictors in the stacking
def plot_pdp(X_train, model, features):
    num_columns = 3
    num_rows = (len(features) + num_columns - 1) // num_columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(28, num_rows * 4))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        PartialDependenceDisplay.from_estimator(model, X_train, [feature], percentiles=(0, 1), ax=axes[i])
    plt.suptitle(f"PDP - {getattr(model, 'name', 'Stacking')}", fontsize=28, weight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
plot_pdp(X_train, stacking_model, X_train.columns)

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
plot_density_scatter(Y_validation, y_pred_RF, model_name="Random Forest")
plot_density_scatter(Y_validation, y_pred_LGBM, model_name="LightGBM")
plot_density_scatter(Y_validation, y_pred_XGB, model_name="XGBoost")
plot_density_scatter(Y_validation, y_pred_stacking, model_name="Stacking")

base_path = r'F:\rapeseed'
output_path = r'F:\pythonProject\SD-yield\predict-yield'
years = [f'{y:02d}' for y in range(1, 21)]

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

        # Stack the predictors
        X = np.stack((Year, DEM, LAI, TN, TP, SIF, Pre, Tmax, Tmin), axis=-1)
        rows, cols, bands = X.shape
        X_2d = X.reshape(rows * cols, bands)

        # Predict by using stacking-EL
        y_pred = StackingRegressor.predict(X_2d)
        out_file = os.path.join(output_path, f'predict{y}_stacking.tif')
        with rasterio.open(out_file, 'w', **meta) as dst:
            dst.write(y_pred.reshape(rows, cols), 1)

        # Predict by using RF
        y_pred = forest.predict(X_2d)
        out_file = os.path.join(output_path, f'predict{y}_RF.tif')
        with rasterio.open(out_file, 'w', **meta) as dst:
            dst.write(y_pred.reshape(rows, cols), 1)

        # Predict by using LGBM
        y_pred = LGBM.predict(X_2d)
        out_file = os.path.join(output_path, f'predict{y}_lgbm.tif')
        with rasterio.open(out_file, 'w', **meta) as dst:
            dst.write(y_pred.reshape(rows, cols), 1)

        # Predict by using XGB
        y_pred = XGB_model.predict(X_2d)
        out_file = os.path.join(output_path, f'predict{y}_xgb.tif')
        with rasterio.open(out_file, 'w', **meta) as dst:
            dst.write(y_pred.reshape(rows, cols), 1)

        print(f"year 20{y} finished！")
    except Exception as e:
        print(f"year 20{y} processing error：{e}")