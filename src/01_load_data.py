
import os
import sys
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "fire_archive_SV-C2_842285.csv"

# Create Spark session
spark = (
    SparkSession.builder
    .appName("FireWatch-AI-Data-Loading")
    .master("local[2]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    print("\n===== FIREWATCH AI: DATA LOADING =====")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    # Load CSV
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(DATA_PATH))
    )

    print("\n1. Total Rows:")
    print(df.count())

    print("\n2. Total Columns:")
    print(len(df.columns))

    print("\n3. Column Names:")
    print(df.columns)

    print("\n4. Dataset Schema:")
    df.printSchema()

    print("\n5. First 5 Records:")
    df.show(5, truncate=False)

    print("\nSUCCESS: NASA FIRMS dataset loaded!")

finally:
    spark.stop()
