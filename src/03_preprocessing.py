
# =========================================================
# FIREWATCH AI
# STEP 3.3 - DATA PREPROCESSING & FEATURE ENGINEERING
# Apache Spark + PyArrow
# =========================================================

import os
import sys
from pathlib import Path

# Configure Python for Spark
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pyarrow as pa
import pyarrow.parquet as pq

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "fire_archive_SV-C2_842285.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "firewatch_cleaned_file.parquet"
)


# =========================================================
# 2. CREATE SPARK SESSION
# =========================================================

spark = (
    SparkSession.builder
    .appName("FireWatch-AI-Preprocessing")
    .master("local[2]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


try:

    print("\n" + "=" * 55)
    print("FIREWATCH AI - DATA PREPROCESSING")
    print("=" * 55)

    # =====================================================
    # 3. LOAD RAW DATA
    # =====================================================

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(INPUT_PATH))
    )

    original_count = df.count()

    print(f"\nOriginal Records: {original_count:,}")
    print(f"Original Columns: {len(df.columns)}")

    # =====================================================
    # 4. DATA CLEANING
    # =====================================================

    print("\n[1] Cleaning Data...")

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "brightness",
            "frp",
            "acq_date"
        ]
    )

    df = df.filter(
        F.col("latitude").between(-90, 90)
        & F.col("longitude").between(-180, 180)
        & (F.col("frp") >= 0)
        & (F.col("brightness") > 0)
    )

    valid_count = df.count()

    print(f"Valid Records: {valid_count:,}")
    print(
        f"Invalid Records Removed: "
        f"{original_count - valid_count:,}"
    )

    # =====================================================
    # 5. FILTER VEGETATION FIRE DETECTIONS
    # =====================================================

    print("\n[2] Filtering Detection Types...")

    # type = 0: Presumed vegetation fire detections
    df = df.filter(F.col("type") == 0)

    vegetation_count = df.count()

    print(
        f"Vegetation Fire Detections: "
        f"{vegetation_count:,}"
    )

    print(
        f"Other Detection Types Excluded: "
        f"{valid_count - vegetation_count:,}"
    )

    # =====================================================
    # 6. FEATURE ENGINEERING
    # =====================================================

    print("\n[3] Creating New Features...")

    df = (
        df
        .withColumn(
            "year",
            F.year("acq_date")
        )
        .withColumn(
            "month",
            F.month("acq_date")
        )
        .withColumn(
            "day_of_year",
            F.dayofyear("acq_date")
        )
    )

    # Analytical FRP categories
    # These are NOT official wildfire severity levels.

    df = df.withColumn(
        "frp_category",
        F.when(
            F.col("frp") < 10,
            "Low"
        )
        .when(
            F.col("frp") < 50,
            "Moderate"
        )
        .otherwise("High")
    )

    print("Features Created Successfully!")

    # =====================================================
    # 7. DATASET SUMMARY
    # =====================================================

    print("\n[4] Final Dataset Summary")

    final_count = df.count()

    print(f"Final Records: {final_count:,}")
    print(f"Final Columns: {len(df.columns)}")

    print("\nFRP Categories:")

    df.groupBy(
        "frp_category"
    ).count().orderBy(
        "frp_category"
    ).show()

    print("\nMonthly Detection Counts:")

    df.groupBy(
        "month"
    ).count().orderBy(
        "month"
    ).show(12)

    # =====================================================
    # 8. SAVE USING PYARROW
    # =====================================================

    print("\n[5] Saving Processed Dataset...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Temporary workaround for Hadoop winutils issue.
    # Collects data into driver RAM.
    # Suitable only while dataset fits in memory.

    pandas_df = df.toPandas()

    arrow_table = pa.Table.from_pandas(
        pandas_df,
        preserve_index=False
    )

    pq.write_table(
        arrow_table,
        str(OUTPUT_PATH),
        compression="snappy"
    )

    print("\nParquet File Saved Successfully!")
    print(f"File Path: {OUTPUT_PATH}")

    # =====================================================
    # 9. VERIFY SAVED FILE
    # =====================================================

    print("\n[6] Verifying Saved File...")

    saved_table = pq.read_table(
        str(OUTPUT_PATH)
    )

    print(
        f"Saved Records: "
        f"{saved_table.num_rows:,}"
    )

    print(
        f"Saved Columns: "
        f"{saved_table.num_columns}"
    )

    if saved_table.num_rows != final_count:
        raise ValueError(
            "Saved row count does not match "
            "the processed dataset!"
        )

    print("\n" + "=" * 55)
    print("SUCCESS: PREPROCESSING COMPLETED!")
    print("=" * 55)


finally:

    spark.stop()
