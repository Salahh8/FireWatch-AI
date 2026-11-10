
# =========================================================
# FIREWATCH AI
# PHASE 5 - MACHINE LEARNING
# GEOGRAPHIC HOTSPOT CLUSTERING USING SPARK MLLIB
# =========================================================

import os
import sys
import json
from pathlib import Path

# Configure Python for Spark
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pyarrow as pa
import pyarrow.parquet as pq

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


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

MODELS_DIR = BASE_DIR / "models"

ANALYTICS_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "analytics"
)

MODEL_METADATA_PATH = (
    MODELS_DIR / "kmeans_metadata.json"
)

CLUSTER_STATS_PATH = (
    ANALYTICS_DIR / "cluster_statistics.parquet"
)


# =========================================================
# 2. CREATE SPARK SESSION
# =========================================================

spark = (
    SparkSession.builder
    .appName("FireWatch-AI-KMeans-Clustering")
    .master("local[2]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


try:

    print("\n" + "=" * 60)
    print("FIREWATCH AI - GEOGRAPHIC HOTSPOT CLUSTERING")
    print("=" * 60)

    # =====================================================
    # 3. LOAD PROCESSED DATA
    # =====================================================

    print("\n[1] LOADING DATASET")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = spark.read.parquet(str(DATA_PATH))

    total_records = df.count()

    print(f"Total Records: {total_records:,}")

    # =====================================================
    # 4. PREPARE GEOGRAPHIC FEATURES
    # =====================================================

    print("\n[2] PREPARING GEOGRAPHIC FEATURES")

    geo_df = (
        df.select(
            "latitude",
            "longitude",
            "frp"
        )
        .dropna(
            subset=["latitude", "longitude", "frp"]
        )
    )

    assembler = VectorAssembler(
        inputCols=["latitude", "longitude"],
        outputCol="features"
    )

    features_df = assembler.transform(geo_df)

    print("Features Prepared: latitude, longitude")

    # =====================================================
    # 5. TRAIN K-MEANS MODEL
    # =====================================================

    K = 4

    print(f"\n[3] TRAINING K-MEANS (K={K})")

    kmeans = KMeans(
        k=K,
        seed=42,
        maxIter=20,
        featuresCol="features",
        predictionCol="cluster"
    )

    model = kmeans.fit(features_df)

    print("Model Training Completed!")

    # =====================================================
    # 6. GENERATE CLUSTER PREDICTIONS
    # =====================================================

    print("\n[4] GENERATING CLUSTER PREDICTIONS")

    predictions = model.transform(features_df)

    predictions.select(
        "latitude",
        "longitude",
        "frp",
        "cluster"
    ).show(10, truncate=False)

    # =====================================================
    # 7. EVALUATE CLUSTERING MODEL
    # =====================================================

    print("\n[5] EVALUATING CLUSTERS")

    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="cluster",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )

    silhouette = evaluator.evaluate(predictions)

    print(f"Silhouette Score: {silhouette:.4f}")

    # =====================================================
    # 8. CLUSTER STATISTICS
    # =====================================================

    print("\n[6] CLUSTER STATISTICS")

    cluster_stats = (
        predictions
        .groupBy("cluster")
        .agg(
            F.count("*").alias("detections"),

            F.round(
                F.avg("latitude"), 4
            ).alias("center_lat"),

            F.round(
                F.avg("longitude"), 4
            ).alias("center_lon"),

            F.round(
                F.avg("frp"), 2
            ).alias("avg_frp")
        )
        .orderBy("cluster")
    )

    cluster_stats.show(truncate=False)

    # =====================================================
    # 9. CLUSTER CENTERS
    # =====================================================

    print("\n[7] K-MEANS CLUSTER CENTERS")

    centers = model.clusterCenters()

    for i, center in enumerate(centers):
        print(
            f"Cluster {i}: "
            f"Latitude={center[0]:.4f}, "
            f"Longitude={center[1]:.4f}"
        )

    # =====================================================
    # 10. CREATE OUTPUT DIRECTORIES
    # =====================================================

    print("\n[8] SAVING CLUSTERING RESULTS")

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ANALYTICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # 11. SAVE MODEL METADATA AS JSON
    # =====================================================

    model_info = {
        "project": "FireWatch AI",
        "algorithm": "Spark MLlib K-Means",
        "k": K,
        "features": [
            "latitude",
            "longitude"
        ],
        "distance_measure": "Euclidean on geographic degrees",
        "seed": 42,
        "max_iterations": 20,
        "total_records": total_records,
        "silhouette_score": float(silhouette),
        "cluster_centers": [
            {
                "cluster": i,
                "latitude": float(center[0]),
                "longitude": float(center[1])
            }
            for i, center in enumerate(centers)
        ]
    }

    with open(
        MODEL_METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            model_info,
            f,
            indent=4
        )

    print(
        "Saved: models/kmeans_metadata.json"
    )

    # =====================================================
    # 12. SAVE CLUSTER STATISTICS AS PARQUET
    # =====================================================

    cluster_pd = cluster_stats.toPandas()

    cluster_table = pa.Table.from_pandas(
        cluster_pd,
        preserve_index=False
    )

    pq.write_table(
        cluster_table,
        str(CLUSTER_STATS_PATH),
        compression="snappy"
    )

    print(
        "Saved: analytics/cluster_statistics.parquet"
    )

    # =====================================================
    # 13. VERIFY SAVED RESULTS
    # =====================================================

    print("\n[9] VERIFYING SAVED FILES")

    saved_stats = pq.read_table(
        str(CLUSTER_STATS_PATH)
    )

    if saved_stats.num_rows != K:
        raise ValueError(
            "Cluster statistics row count "
            "does not match K!"
        )

    with open(
        MODEL_METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        saved_metadata = json.load(f)

    if saved_metadata["k"] != K:
        raise ValueError(
            "Saved model metadata does not match K!"
        )

    print(
        f"Verified Cluster Statistics: "
        f"{saved_stats.num_rows} clusters"
    )

    print(
        f"Verified Model Metadata: K={saved_metadata['k']}"
    )

    # =====================================================
    # 14. FINAL MESSAGE
    # =====================================================

    print("\n" + "=" * 60)
    print("SUCCESS: K-MEANS CLUSTERING COMPLETED!")
    print("MODEL METADATA AND STATISTICS SAVED!")
    print("=" * 60)


finally:
    spark.stop()
