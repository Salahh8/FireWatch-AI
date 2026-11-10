
import os
import sys
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "fire_archive_SV-C2_842285.csv"

spark = (
    SparkSession.builder
    .appName("FireWatch-AI-Data-Quality")
    .master("local[2]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(DATA_PATH))
    )

    print("\n===== FIREWATCH AI: DATA QUALITY =====")

    # 1. Total records
    total_rows = df.count()
    print(f"\nTotal Rows: {total_rows:,}")

    # 2. Missing values
    print("\n===== MISSING VALUES =====")

    missing_exprs = [
        F.sum(
            F.when(F.col(c).isNull(), 1).otherwise(0)
        ).alias(c)
        for c in df.columns
    ]

    df.select(missing_exprs).show(truncate=False)

    # 3. Exact duplicate records
    print("\n===== DUPLICATES =====")
    unique_rows = df.dropDuplicates().count()
    print(f"Duplicate Rows: {total_rows - unique_rows:,}")

    # 4. Numeric statistics
    print("\n===== NUMERIC STATISTICS =====")
    df.select(
        "latitude",
        "longitude",
        "brightness",
        "frp",
        "scan",
        "track"
    ).describe().show(truncate=False)

    # 5. Date range
    print("\n===== DATE RANGE =====")
    df.select(
        F.min("acq_date").alias("first_date"),
        F.max("acq_date").alias("last_date")
    ).show()

    # 6. Confidence distribution
    print("\n===== CONFIDENCE DISTRIBUTION =====")
    df.groupBy("confidence").count().orderBy("confidence").show()

    # 7. Day vs Night
    print("\n===== DAY / NIGHT =====")
    df.groupBy("daynight").count().show()

    # 8. Detection types
    print("\n===== DETECTION TYPES =====")
    df.groupBy("type").count().orderBy("type").show()

    print("\nSUCCESS: Data quality assessment completed!")

finally:
    spark.stop()
