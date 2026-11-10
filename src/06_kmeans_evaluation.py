
import os
import sys
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "firewatch_cleaned_file.parquet"
)

spark = (
    SparkSession.builder
    .appName("FireWatch-KMeans-Evaluation")
    .master("local[2]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    print("\nFIREWATCH AI - K-MEANS EVALUATION")

    df = spark.read.parquet(str(DATA_PATH))

    # Use a reproducible sample for faster comparison.
    sample_df = (
        df.select("latitude", "longitude")
        .dropna()
        .sample(
            withReplacement=False,
            fraction=0.10,
            seed=42
        )
    )

    assembler = VectorAssembler(
        inputCols=["latitude", "longitude"],
        outputCol="features"
    )

    features_df = assembler.transform(sample_df).cache()

    sample_count = features_df.count()
    print(f"\nSample Records: {sample_count:,}")

    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="cluster",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )

    print("\nK-MEANS COMPARISON")
    print("-" * 35)

    results = []

    for k in [3, 4, 5, 6]:

        print(f"\nTraining K-Means with k={k}...")

        kmeans = KMeans(
            k=k,
            seed=42,
            maxIter=20,
            featuresCol="features",
            predictionCol="cluster"
        )

        model = kmeans.fit(features_df)

        predictions = model.transform(features_df)

        score = evaluator.evaluate(predictions)

        results.append((k, score))

        print(f"k={k} | Silhouette={score:.4f}")

    best_k, best_score = max(
        results,
        key=lambda x: x[1]
    )

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)

    for k, score in results:
        print(f"k={k}: {score:.4f}")

    print(f"\nBest K (sample): {best_k}")
    print(f"Best Silhouette (sample): {best_score:.4f}")

    print("\nSUCCESS: EVALUATION COMPLETED!")

    features_df.unpersist()

finally:
    spark.stop()
