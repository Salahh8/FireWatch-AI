
# =========================================================
# FIREWATCH AI - SPARK GEOGRAPHIC JOIN
# =========================================================

import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "firewatch_cleaned_file.parquet"
)

OUTPUT_DIR = (
    BASE_DIR / "data" / "processed" / "analytics"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("FireWatch-AI-Geographic-Join")
    .master("local[2]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    print("\n" + "=" * 55)
    print("FIREWATCH AI - SPARK GEOGRAPHIC JOIN")
    print("=" * 55)

    print("\n[1] Loading Satellite Data...")

    df = spark.read.parquet(str(DATA_PATH))

    print(f"Total Records: {df.count():,}")

    print("\n[2] Creating Geographic Grid Keys...")

    detections = (
        df.withColumn(
            "lat_grid",
            F.floor(F.col("latitude"))
        )
        .withColumn(
            "lon_grid",
            F.floor(F.col("longitude"))
        )
    )

    print("\n[3] Creating Geographic Reference Table...")

    grid_reference = (
        detections
        .select("lat_grid", "lon_grid")
        .distinct()
        .withColumn(
            "region_id",
            F.concat(
                F.lit("GRID_"),
                F.col("lat_grid").cast("string"),
                F.lit("_"),
                F.col("lon_grid").cast("string")
            )
        )
    )

    print(
        f"Geographic Grid Regions: "
        f"{grid_reference.count():,}"
    )

    print("\n[4] Performing Spark LEFT JOIN...")

    enriched = detections.join(
        F.broadcast(grid_reference),
        on=["lat_grid", "lon_grid"],
        how="left"
    )

    print("\n[5] Validating Join...")

    original_count = detections.count()
    joined_count = enriched.count()

    missing_regions = enriched.filter(
        F.col("region_id").isNull()
    ).count()

    print(f"Original Records: {original_count:,}")
    print(f"Joined Records: {joined_count:,}")
    print(f"Missing Region IDs: {missing_regions:,}")

    if original_count != joined_count:
        raise ValueError(
            "Join validation failed: row counts differ!"
        )

    if missing_regions != 0:
        raise ValueError(
            "Join validation failed: missing region IDs!"
        )

    print("\n[6] Aggregating Regional Statistics...")

    region_stats = (
        enriched
        .groupBy(
            "region_id",
            "lat_grid",
            "lon_grid"
        )
        .agg(
            F.count("*").alias("detections"),
            F.round(
                F.avg("frp"), 2
            ).alias("avg_frp"),
            F.round(
                F.max("frp"), 2
            ).alias("max_frp")
        )
        .orderBy(
            F.desc("detections")
        )
    )

    region_stats.show(10, truncate=False)

    print("\n[7] Saving Regional Statistics...")

    output_path = (
        OUTPUT_DIR / "regional_join_statistics.parquet"
    )

    region_pd = region_stats.toPandas()

    pq.write_table(
        pa.Table.from_pandas(
            region_pd,
            preserve_index=False
        ),
        str(output_path),
        compression="snappy"
    )

    print(
        "Saved: analytics/regional_join_statistics.parquet"
    )

    print("\n" + "=" * 55)
    print("SUCCESS: SPARK JOIN COMPLETED!")
    print("=" * 55)

finally:
    spark.stop()
