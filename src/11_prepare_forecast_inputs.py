
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]

DATA_PATH = BASE / "data/processed/firewatch_cleaned_file.parquet"
GRID_PATH = BASE / "data/processed/australia_grid.parquet"
OUTPUT_PATH = BASE / "data/processed/forecast_inputs_2025_01.parquet"

print("\n=== LOADING DATA ===")

df = pd.read_parquet(DATA_PATH)
grid = pd.read_parquet(GRID_PATH)

required = {"latitude", "longitude", "acq_date"}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing detection columns: {missing}")

if not {"lat_grid", "lon_grid"}.issubset(grid.columns):
    raise ValueError("Grid columns missing")

df["acq_date"] = pd.to_datetime(df["acq_date"])
df["month"] = df["acq_date"].dt.month

df["lat_grid"] = np.floor(df["latitude"]).astype(int)
df["lon_grid"] = np.floor(df["longitude"]).astype(int)

print("Detections loaded:", len(df))
print("Grid cells:", len(grid))

print("\n=== AGGREGATING LAST THREE MONTHS ===")

recent = df[
    (df["acq_date"].dt.year == 2024)
    & (df["month"].isin([10, 11, 12]))
]

counts = (
    recent.groupby(["lat_grid", "lon_grid", "month"])
    .size()
    .reset_index(name="detections")
)

pivot = counts.pivot_table(
    index=["lat_grid", "lon_grid"],
    columns="month",
    values="detections",
    fill_value=0
)

pivot = pivot.reindex(columns=[10, 11, 12], fill_value=0)

pivot = pivot.rename(
    columns={
        12: "previous_1_month_detections",
        11: "previous_2_month_detections",
        10: "previous_3_month_detections"
    }
).reset_index()

forecast = grid[["lat_grid", "lon_grid"]].merge(
    pivot,
    on=["lat_grid", "lon_grid"],
    how="left"
)

lag_columns = [
    "previous_1_month_detections",
    "previous_2_month_detections",
    "previous_3_month_detections"
]

forecast[lag_columns] = (
    forecast[lag_columns].fillna(0).astype(int)
)

forecast["month"] = 1
forecast["forecast_year"] = 2025

forecast = forecast[
    [
        "lat_grid",
        "lon_grid",
        "month",
        "forecast_year",
        *lag_columns
    ]
]

if forecast[["lat_grid", "lon_grid"]].duplicated().any():
    raise ValueError("Duplicate grid cells")

if forecast[lag_columns].isna().any().any():
    raise ValueError("Missing lag values")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
forecast.to_parquet(OUTPUT_PATH, index=False)

print("\n=== FORECAST DATASET ===")
print("Rows:", len(forecast))
print("Forecast month: January 2025")
print("Unique grid cells:", forecast[["lat_grid", "lon_grid"]].drop_duplicates().shape[0])

print("\nSample:")
print(forecast.head(5).to_string(index=False))

print("\nSaved:", OUTPUT_PATH)
print("\nFORECAST_INPUTS_SUCCESS")
