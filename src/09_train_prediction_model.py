
from pathlib import Path
import os
import sys
import json
import traceback

BASE = Path(__file__).resolve().parents[1]

# Fix Python worker configuration on Windows
PYTHON_EXE = sys.executable
os.environ["PYSPARK_PYTHON"] = PYTHON_EXE
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_EXE

import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

INPUT = BASE / "data/processed/prediction_dataset.parquet"
MODEL_DIR = BASE / "models/geographic_rf"
METRICS_FILE = BASE / "models/geographic_rf_metrics.json"

spark = (
    SparkSession.builder
    .appName("FireWatch-Geographic-RandomForest")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\n=== LOADING PREDICTION DATA ===")

pdf = pd.read_parquet(INPUT)

features = [
    "lat_grid",
    "lon_grid",
    "month",
    "lag_1",
    "lag_2",
    "lag_3"
]

# Exclude current-month detections from features
columns = features + ["target"]

df = spark.createDataFrame(pdf[columns])

assembler = VectorAssembler(
    inputCols=features,
    outputCol="features"
)

df = assembler.transform(df)

train = df.filter(F.col("month") <= 9)
validation = df.filter(F.col("month") == 10)
test = df.filter(F.col("month") >= 11)

print("Train rows:", train.count())
print("Validation rows:", validation.count())
print("Test rows:", test.count())

print("\n=== TRAINING RANDOM FOREST ===")

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="target",
    numTrees=100,
    maxDepth=8,
    seed=42
)

model = rf.fit(train)

print("Training complete.")

def evaluate(data, name):
    predictions = model.transform(data)

    accuracy = MulticlassClassificationEvaluator(
        labelCol="target",
        predictionCol="prediction",
        metricName="accuracy"
    ).evaluate(predictions)

    precision = MulticlassClassificationEvaluator(
        labelCol="target",
        predictionCol="prediction",
        metricName="weightedPrecision"
    ).evaluate(predictions)

    recall = MulticlassClassificationEvaluator(
        labelCol="target",
        predictionCol="prediction",
        metricName="weightedRecall"
    ).evaluate(predictions)

    f1 = MulticlassClassificationEvaluator(
        labelCol="target",
        predictionCol="prediction",
        metricName="f1"
    ).evaluate(predictions)

    auc = BinaryClassificationEvaluator(
        labelCol="target",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    ).evaluate(predictions)

    result = {
        "accuracy": round(accuracy, 4),
        "weighted_precision": round(precision, 4),
        "weighted_recall": round(recall, 4),
        "weighted_f1": round(f1, 4),
        "roc_auc": round(auc, 4)
    }

    print(f"\n=== {name.upper()} RESULTS ===")
    for key, value in result.items():
        print(f"{key}: {value}")

    return result

validation_metrics = evaluate(validation, "Validation")
test_metrics = evaluate(test, "Test")

print("\n=== FEATURE IMPORTANCE ===")

for name, importance in zip(
    features, model.featureImportances.toArray()
):
    print(f"{name}: {importance:.4f}")


MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

metrics = {
    "model": "Spark MLlib Random Forest",
    "training_months": [4, 5, 6, 7, 8, 9],
    "validation_months": [10],
    "testing_months": [11, 12],
    "features": features,
    "validation": validation_metrics,
    "test": test_metrics,
    "feature_importance": {
        name: round(float(value), 6)
        for name, value in zip(
            features, model.featureImportances.toArray()
        )
    }
}

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("\nMetrics saved:", METRICS_FILE)

try:
    model.write().overwrite().save(str(MODEL_DIR))
    print("Spark model saved:", MODEL_DIR)
except Exception:
    print("\n=== MODEL SAVE ERROR ===")
    traceback.print_exc()

print("\n=== COMPLETE ===")

spark.stop()

