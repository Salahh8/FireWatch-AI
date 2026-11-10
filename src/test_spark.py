import os
import sys

python_path = sys.executable

os.environ["PYSPARK_PYTHON"] = python_path
os.environ["PYSPARK_DRIVER_PYTHON"] = python_path
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

print("=" * 50)
print("FireWatch AI - Apache Spark Test")
print("=" * 50)


spark = (
    SparkSession.builder
    .appName("FireWatch-AI-Test")
    .master("local[1]")
    .config("spark.ui.enabled", "false")
    .config("spark.python.worker.reuse", "false")
    .config("spark.python.worker.faulthandler.enabled", "true")
    .getOrCreate()
)


spark.sparkContext.setLogLevel("ERROR")

print("Python:", sys.version.split()[0])
print("Spark:", spark.version)

data = [
    ("Australia", 150, 25.5),
    ("Brazil", 200, 42.3),
    ("Canada", 100, 18.7),
]

df = spark.createDataFrame(
    data,
    ["country", "hotspots", "avg_frp"]
)

print("\nOriginal DataFrame:")
df.show()

print("\nFiltered DataFrame:")
df.filter(col("hotspots") > 100).show()

print("\nTotal Hotspots:")
df.groupBy().sum("hotspots").show()

spark.stop()

print("\nSUCCESS: Apache Spark is working!")
