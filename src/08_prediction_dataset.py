
from pathlib import Path
import os
import sys

# Ensure Spark uses the current virtual environment.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Local Windows Spark configuration.
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ==========================================================
# FIREWATCH AI - GEOGRAPHIC PREDICTION DATASET
# ==========================================================

BASE = Path(__file__).resolve().parents[1]

CLEANED_PATH = (
    BASE / "data" / "processed"
    / "firewatch_cleaned_file.parquet"
)

GRID_PATH = (
    BASE / "data" / "processed"
    / "australia_grid.parquet"
)

OUTPUT_PATH = (
    BASE / "data" / "processed"
    / "prediction_dataset.parquet"
)

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("FireWatch-Prediction-Dataset")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    print("\n=== 1. LOADING DATA ===")

    fires = spark.read.parquet(str(CLEANED_PATH))

    grid_pdf = pd.read_parquet(GRID_PATH)

    required_columns = {"lat_grid", "lon_grid"}

    if not required_columns.issubset(grid_pdf.columns):
        raise ValueError("Invalid Australia grid columns.")

    grid_pdf = (
        grid_pdf[["lat_grid", "lon_grid"]]
        .drop_duplicates()
        .copy()
    )

    grid = spark.createDataFrame(grid_pdf)

    grid = grid.select(
        F.col("lat_grid").cast("int"),
        F.col("lon_grid").cast("int")
    )

    print("Fire detections:", fires.count())
    print("Australian grid cells:", grid.count())

    print("\n=== 2. ASSIGNING GRID CELLS ===")

    fires = (
        fires
        .withColumn(
            "lat_grid",
            F.floor("latitude").cast("int")
        )
        .withColumn(
            "lon_grid",
            F.floor("longitude").cast("int")
        )
    )

    # Check whether all detected coordinates fit the new grid.
    outside_count = (
        fires.join(
            grid,
            ["lat_grid", "lon_grid"],
            "left_anti"
        )
        .count()
    )

    print("Detections outside selected grid:", outside_count)

    if outside_count > 0:
        print(
            "WARNING: Some detections do not match the "
            "selected geographic boundary."
        )

    fires = fires.join(
        grid,
        ["lat_grid", "lon_grid"],
        "inner"
    )

    print("Detections inside grid:", fires.count())

    print("\n=== 3. CREATING REGION-MONTH GRID ===")

    months = (
        spark.range(1, 13)
        .select(F.col("id").cast("int").alias("month"))
    )

    all_combinations = grid.crossJoin(months)

    monthly_counts = (
        fires
        .groupBy("lat_grid", "lon_grid", "month")
        .agg(F.count("*").alias("detections"))
    )

    dataset = (
        all_combinations
        .join(
            monthly_counts,
            ["lat_grid", "lon_grid", "month"],
            "left"
        )
        .fillna({"detections": 0})
    )

    print("\n=== 4. CREATING HISTORICAL FEATURES ===")

    window = (
        Window
        .partitionBy("lat_grid", "lon_grid")
        .orderBy("month")
    )

    dataset = (
        dataset
        .withColumn(
            "lag_1",
            F.lag("detections", 1).over(window)
        )
        .withColumn(
            "lag_2",
            F.lag("detections", 2).over(window)
        )
        .withColumn(
            "lag_3",
            F.lag("detections", 3).over(window)
        )
    )

    # January-March are used as historical context.
    dataset = dataset.filter(F.col("month") >= 4)

    dataset = dataset.withColumn(
        "target",
        F.when(
            F.col("detections") > 0, 1
        ).otherwise(0)
    )

    dataset = dataset.select(
        "lat_grid",
        "lon_grid",
        "month",
        "lag_1",
        "lag_2",
        "lag_3",
        "target",
        "detections"
    )

    # Check that historical features are available.
    missing = (
        dataset
        .filter(
            F.col("lag_1").isNull()
            | F.col("lag_2").isNull()
            | F.col("lag_3").isNull()
        )
        .count()
    )

    if missing != 0:
        raise ValueError(
            f"Found {missing} rows with missing lag features."
        )

    print("\n=== 5. VALIDATING DATASET ===")

    total_rows = dataset.count()
    grid_rows = grid.count()

    expected_rows = grid_rows * 9

    print("Expected rows:", expected_rows)
    print("Actual rows:", total_rows)

    if total_rows != expected_rows:
        raise ValueError("Unexpected prediction dataset size.")

    dataset.groupBy("target").count().show()

    print("\n=== 6. SAVING DATASET ===")

    # Using Pandas/PyArrow for small final output
    # avoids the previous Windows Spark write issue.
    pdf = dataset.toPandas()

    pdf = pdf.sort_values(
        ["lat_grid", "lon_grid", "month"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    pdf.to_parquet(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", pdf.shape)

    print("\n=== TEMPORAL SPLIT SIZES ===")

    print(
        "Train (Apr-Sep):",
        len(pdf[pdf["month"].between(4, 9)])
    )

    print(
        "Validation (Oct):",
        len(pdf[pdf["month"] == 10])
    )

    print(
        "Test (Nov-Dec):",
        len(pdf[pdf["month"].between(11, 12)])
    )

    print("\n=== COMPLETE ===")

finally:
    spark.stop()
