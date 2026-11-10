
# =========================================================
# FIREWATCH AI
# PHASE 4 - EXPLORATORY DATA ANALYSIS
# Apache Spark DataFrames + Spark SQL
# =========================================================

import os
import sys
from pathlib import Path

# Configure Python for PySpark
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

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "firewatch_cleaned_file.parquet"
)

ANALYTICS_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "analytics"
)


# =========================================================
# 2. HELPER FUNCTION: EXPORT PARQUET
# =========================================================

def export_parquet(spark_df, filename):
    """
    Export aggregated Spark DataFrames to Parquet
    using PyArrow to avoid Windows winutils issues.

    Only use this for small aggregated datasets.
    """

    output_path = ANALYTICS_DIR / filename

    pandas_df = spark_df.toPandas()

    table = pa.Table.from_pandas(
        pandas_df,
        preserve_index=False
    )

    pq.write_table(
        table,
        str(output_path),
        compression="snappy"
    )

    print(
        f"Saved: {filename} "
        f"({len(pandas_df):,} rows)"
    )


# =========================================================
# 3. CREATE SPARK SESSION
# =========================================================

spark = (
    SparkSession.builder
    .appName("FireWatch-AI-EDA")
    .master("local[2]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


try:

    print("\n" + "=" * 60)
    print("FIREWATCH AI - SPARK EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # =====================================================
    # 4. LOAD PROCESSED DATA
    # =====================================================

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Parquet file not found: {DATA_PATH}"
        )

    df = spark.read.parquet(str(DATA_PATH))

    print("\n[1] DATASET OVERVIEW")

    total_rows = df.count()

    print(f"Total Records: {total_rows:,}")
    print(f"Total Columns: {len(df.columns)}")

    print("\nDataset Schema:")
    df.printSchema()

    print("\nFirst 5 Records:")
    df.show(5, truncate=False)

    # =====================================================
    # 5. MONTHLY DETECTION COUNTS
    # =====================================================

    print("\n[2] MONTHLY HOTSPOT DETECTIONS")

    monthly_counts = (
        df.groupBy("month")
        .agg(
            F.count("*").alias("total_detections")
        )
        .orderBy("month")
    )

    monthly_counts.show(12, truncate=False)

    # =====================================================
    # 6. REGISTER SPARK SQL VIEW
    # =====================================================

    print("\n[3] REGISTERING SPARK SQL VIEW")

    df.createOrReplaceTempView("fire_detections")

    print("SQL View Created: fire_detections")

    # =====================================================
    # 7. MONTHLY FRP ANALYSIS
    # =====================================================

    print("\n[4] MONTHLY FRP ANALYSIS")

    monthly_frp = spark.sql("""
        SELECT
            month,
            COUNT(*) AS total_detections,
            ROUND(AVG(frp), 2) AS avg_frp,
            ROUND(MAX(frp), 2) AS max_frp,
            ROUND(SUM(frp), 2) AS total_frp
        FROM fire_detections
        GROUP BY month
        ORDER BY month
    """)

    monthly_frp.show(12, truncate=False)

    # =====================================================
    # 8. DAY VS NIGHT ANALYSIS
    # =====================================================

    print("\n[5] DAY VS NIGHT ANALYSIS")

    daynight_analysis = spark.sql("""
        SELECT
            daynight,
            COUNT(*) AS detections,
            ROUND(AVG(frp), 2) AS avg_frp,
            ROUND(MAX(frp), 2) AS max_frp
        FROM fire_detections
        GROUP BY daynight
        ORDER BY daynight
    """)

    daynight_analysis.show(truncate=False)

    # =====================================================
    # 9. CONFIDENCE LEVEL ANALYSIS
    # =====================================================

    print("\n[6] CONFIDENCE LEVEL ANALYSIS")

    confidence_analysis = spark.sql("""
        SELECT
            confidence,
            COUNT(*) AS detections,
            ROUND(AVG(frp), 2) AS avg_frp
        FROM fire_detections
        GROUP BY confidence
        ORDER BY confidence
    """)

    confidence_analysis.show(truncate=False)

    # =====================================================
    # 10. OVERALL FRP STATISTICS
    # =====================================================

    print("\n[7] OVERALL FRP STATISTICS")

    frp_stats = df.select(
        F.round(
            F.avg("frp"), 2
        ).alias("avg_frp"),

        F.round(
            F.max("frp"), 2
        ).alias("max_frp"),

        F.round(
            F.min("frp"), 2
        ).alias("min_frp"),

        F.round(
            F.stddev("frp"), 2
        ).alias("stddev_frp")
    )

    frp_stats.show(truncate=False)

    # =====================================================
    # 11. FRP CATEGORY DISTRIBUTION
    # =====================================================

    print("\n[8] FRP CATEGORY DISTRIBUTION")

    frp_categories = (
        df.groupBy("frp_category")
        .agg(
            F.count("*").alias("detections")
        )
        .orderBy("frp_category")
    )

    frp_categories.show(truncate=False)

    # =====================================================
    # 12. GEOGRAPHIC HOTSPOT ANALYSIS
    # =====================================================

    print("\n[9] GEOGRAPHIC HOTSPOT ANALYSIS")

    # 1-degree latitude/longitude grid.
    # Physical cell sizes vary by latitude.

    geo_df = (
        df
        .withColumn(
            "lat_grid",
            F.floor(F.col("latitude"))
        )
        .withColumn(
            "lon_grid",
            F.floor(F.col("longitude"))
        )
    )

    geographic_hotspots = (
        geo_df
        .groupBy("lat_grid", "lon_grid")
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

    print("\nTop 10 Geographic Hotspot Cells:")

    geographic_hotspots.show(
        10,
        truncate=False
    )

    # =====================================================
    # 13. EXPORT ANALYTICS FOR DASHBOARD
    # =====================================================

    print("\n[10] EXPORTING DASHBOARD DATA")

    ANALYTICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    export_parquet(
        monthly_frp,
        "monthly_statistics.parquet"
    )

    export_parquet(
        geographic_hotspots,
        "geographic_hotspots.parquet"
    )

    export_parquet(
        daynight_analysis,
        "daynight_statistics.parquet"
    )

    export_parquet(
        confidence_analysis,
        "confidence_statistics.parquet"
    )

    export_parquet(
        frp_categories,
        "frp_categories.parquet"
    )

    # =====================================================
    # 14. FINAL MESSAGE
    # =====================================================

    print("\n" + "=" * 60)
    print("SUCCESS: SPARK EDA COMPLETED!")
    print("DASHBOARD ANALYTICS EXPORTED SUCCESSFULLY!")
    print("=" * 60)


finally:
    spark.stop()
