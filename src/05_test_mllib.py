
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

spark = (
    SparkSession.builder
    .appName("FireWatch-MLlib-Test")
    .master("local[2]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    print("\nTesting Spark MLlib...")

    data = [
        (-20.0, 134.0),
        (-20.2, 134.1),
        (-14.0, 132.0),
        (-14.2, 132.1),
        (-26.0, 147.0),
        (-26.1, 147.2)
    ]

    df = spark.createDataFrame(
        data,
        ["latitude", "longitude"]
    )

    assembler = VectorAssembler(
        inputCols=["latitude", "longitude"],
        outputCol="features"
    )

    features_df = assembler.transform(df)

    model = KMeans(
        k=3,
        seed=42,
        featuresCol="features",
        predictionCol="cluster"
    ).fit(features_df)

    predictions = model.transform(features_df)

    predictions.select(
        "latitude",
        "longitude",
        "cluster"
    ).show()

    print("\nSUCCESS: Spark MLlib is working!")

finally:
    spark.stop()
