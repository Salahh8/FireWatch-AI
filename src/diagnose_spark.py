
import os
import sys
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = (
    SparkSession.builder
    .appName("FireWatch-Diagnostic")
    .master("local[1]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    print("Python executable:", sys.executable)
    print("Spark version:", spark.version)

    print("\nTEST 1: Spark SQL without Python worker")
    spark.sql("SELECT 1 AS test_value").show()
    print("TEST 1 PASSED")

    print("\nTEST 2: Python RDD worker")
    result = spark.sparkContext.parallelize([1, 2, 3], 1).map(
        lambda x: x * 2
    ).collect()
    print("Result:", result)
    print("TEST 2 PASSED")

finally:
    spark.stop()
