import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import rasterio
import os
import glob

# Reading the data, construct the quantile to match predict yield and statistical yield
excel_file = r'/SD-yield/predict-yield-250914/resample-1km/mask/excel/statistical-yield-match.xlsx'
df = pd.read_excel(excel_file, engine='openpyxl')

df = df.dropna(subset=['statistical_yield', 'zonal-yield__stacking_2001-2020'])
yield_data = df['statistical_yield'].values
predict_data = df['zonal-yield__stacking_2001-2020'].values

yield_quantiles = np.percentile(yield_data, np.linspace(0, 100, 101))
predict_quantiles = np.percentile(predict_data, np.linspace(0, 100, 101))
f = interp1d(predict_quantiles, yield_quantiles, kind='linear', fill_value="extrapolate")

print(f"[INFO] Yield quantile range: {yield_quantiles[0]} ~ {yield_quantiles[-1]}")
print(f"[INFO] Predict quantile range: {predict_quantiles[0]} ~ {predict_quantiles[-1]}")

# Input data path
input_dir = r'/SD-yield/predict-yield-250914/resample-1km/stacking'
output_dir = r'/SD-yield/predict-yield-250914/resample-1km/stacking/qm'

os.makedirs(output_dir, exist_ok=True)

tif_files = glob.glob(os.path.join(input_dir, '*.tif'))
print(f"[INFO] Found {len(tif_files)} TIF files.")

# iteration of each raster and apply Quantile Mapping
for tif_file in tif_files:
    filename = os.path.basename(tif_file)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_QM{ext}")

    print(f"\n[PROCESSING] {filename} → {os.path.basename(output_path)}")

    with rasterio.open(tif_file) as src:
        data = src.read(1)
        nodata_value = src.nodata

        if nodata_value is not None:
            data = np.where(data == nodata_value, np.nan, data)

        valid_mask = (data > 0) & (data < 10000)
        data_clipped = np.where(valid_mask, np.clip(data, predict_quantiles[0], predict_quantiles[-1]), np.nan)

        corrected_data = np.where(np.isnan(data_clipped), np.nan, f(data_clipped))

        # Set NaN as nodata value of -9999
        corrected_data_out = np.where(np.isnan(corrected_data), -9999, corrected_data)

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=src.height,
            width=src.width,
            count=1,
            dtype='float32',
            crs=src.crs,
            transform=src.transform,
            nodata=-9999
        ) as dst:
            dst.write(corrected_data_out.astype(np.float32), 1)

    print(f"[SAVED] {output_path}")

print("\n All raster files processed and saved.")

