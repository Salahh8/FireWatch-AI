
import os
import sys
import json
from pathlib import Path

# Configure Python workers before importing/starting Spark.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.functions import vector_to_array

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

BASE = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE / "data" / "processed"
    / "prediction_dataset.parquet"
)

RESULTS_DIR = BASE / "reports" / "prediction"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "lat_grid",
    "lon_grid",
    "month",
    "previous_1_month_detections",
    "previous_2_month_detections",
    "previous_3_month_detections",
]

spark = (
    SparkSession.builder
    .master("local[4]")
    .appName("FireWatch-RandomForest-Prediction")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


def calculate_metrics(y_true, y_pred, probabilities=None):
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "confusion_matrix": (
            confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            ).tolist()
        ),
    }

    if probabilities is not None and len(set(y_true)) == 2:
        result["roc_auc"] = float(
            roc_auc_score(y_true, probabilities)
        )
        result["average_precision"] = float(
            average_precision_score(y_true, probabilities)
        )

    return result


def evaluate(model, data, name):
    predictions = model.transform(data)

    predictions = predictions.select(
        "lat_grid",
        "lon_grid",
        "month",
        "label",
        "previous_1_month_detections",
        "prediction",
        vector_to_array("probability")[1].alias(
            "hotspot_probability"
        ),
    )

    pdf = predictions.toPandas()

    y_true = pdf["label"].astype(int)
    y_pred = pdf["prediction"].astype(int)
    probs = pdf["hotspot_probability"]

    metrics = calculate_metrics(y_true, y_pred, probs)

    # Baseline: predict a detection if the previous
    # month had one or more detections.
    baseline_pred = (pdf["previous_1_month_detections"] > 0).astype(int)

    baseline = calculate_metrics(y_true, baseline_pred)

    print(f"\n=== {name.upper()} RESULTS ===")
    print("Records:", len(pdf))

    for key in [
        "accuracy", "precision", "recall",
        "f1", "roc_auc", "average_precision"
    ]:
        if key in metrics:
            print(f"{key}: {metrics[key]:.4f}")

    print("Confusion matrix:", metrics["confusion_matrix"])
    print(f"Persistence baseline F1: {baseline['f1']:.4f}")

    return pdf, {
        "random_forest": metrics,
        "persistence_baseline": baseline,
        "records": len(pdf),
    }


try:
    print("\n=== 1. LOADING DATASET ===")

    df = spark.read.parquet(str(DATA_PATH))

    print("Total records:", df.count())

    required = set(FEATURES + ["target"])
    if not required.issubset(df.columns):
        raise ValueError("Missing required prediction columns.")

    for col_name in FEATURES:
        df = df.withColumn(
            col_name,
            F.col(col_name).cast("double")
        )

    df = df.withColumn(
        "label",
        F.col("target").cast("double")
    )

    assembler = VectorAssembler(
        inputCols=FEATURES,
        outputCol="features",
        handleInvalid="error",
    )

    data = assembler.transform(df).select(
        "lat_grid",
        "lon_grid",
        "month",
        "previous_1_month_detections",
        "features",
        "label",
    )

    print("\n=== 2. TEMPORAL SPLIT ===")

    train = data.filter(F.col("month") <= 9).cache()
    validation = data.filter(F.col("month") == 10).cache()
    test = data.filter(F.col("month") >= 11).cache()

    print("Train:", train.count())
    print("Validation:", validation.count())
    print("Test:", test.count())

    if (
        train.count() != 4932
        or validation.count() != 822
        or test.count() != 1644
    ):
        raise ValueError("Unexpected temporal split sizes.")

    print("\n=== 3. TRAINING RANDOM FOREST ===")

    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=100,
        maxDepth=6,
        maxBins=32,
        seed=42,
    )

    model = rf.fit(train)

    print("Random Forest training complete.")
    print("Number of trees:", model.getNumTrees)

    print("\n=== 4. EVALUATION ===")

    val_pdf, val_metrics = evaluate(
        model, validation, "Validation October 2024"
    )

    test_pdf, test_metrics = evaluate(
        model, test, "Test November-December 2024"
    )

    print("\n=== 5. FEATURE IMPORTANCE ===")

    importances = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.featureImportances.toArray()
    }).sort_values("importance", ascending=False)

    print(importances.to_string(index=False))

    print("\n=== 6. SAVING RESULTS ===")

    results = {
        "project": "FireWatch AI",
        "model": "Spark MLlib Random Forest",
        "target": "At least one satellite hotspot detection in a grid-month",
        "study_year": 2024,
        "grid_size_degrees": 1,
        "features": FEATURES,
        "parameters": {
            "numTrees": 100,
            "maxDepth": 6,
            "seed": 42,
        },
        "validation": val_metrics,
        "test": test_metrics,
        "limitations": [
            "Only one year of observations",
            "No evaluation on another year",
            "No detection is not proof of no fire",
            "Predictions represent monthly hotspot detection",
            "The reported test evaluates rolling monthly forecasts"
        ],
    }

    with open(
        RESULTS_DIR / "prediction_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    importances.to_csv(
        RESULTS_DIR / "feature_importance.csv",
        index=False,
    )

    test_pdf.to_parquet(
        RESULTS_DIR / "test_predictions.parquet",
        index=False,
    )

    print("Saved prediction_metrics.json")
    print("Saved feature_importance.csv")
    print("Saved test_predictions.parquet")

    print("\n=== COMPLETE ===")

finally:
    spark.stop()
