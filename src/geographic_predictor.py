
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models/geographic_rf_export.json"

FEATURES = [
    "lat_grid",
    "lon_grid",
    "month",
    "previous_1_month_detections",
    "previous_2_month_detections",
    "previous_3_month_detections",
]


class GeographicPredictor:
    def __init__(self, model_path=MODEL_PATH):
        with open(model_path, "r", encoding="utf-8") as f:
            self.model = json.load(f)

        if self.model["features"] != FEATURES:
            raise ValueError("Unexpected model features")

        self.trees = self.model["trees"]
        self.weights = self.model["tree_weights"]

        if len(self.trees) != len(self.weights):
            raise ValueError("Tree/weight mismatch")

    @staticmethod
    def predict_tree(node, values):
        while node["type"] == "split":
            feature_value = values[node["feature_index"]]

            if feature_value <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]

        counts = np.asarray(node["class_counts"], dtype=float)
        total = counts.sum()

        if total <= 0:
            raise ValueError("Invalid leaf statistics")

        return counts / total

    def predict(self, lat_grid, lon_grid, month, previous_1,
                previous_2, previous_3):

        values = np.asarray(
            [
                lat_grid,
                lon_grid,
                month,
                previous_1,
                previous_2,
                previous_3,
            ],
            dtype=float,
        )

        probabilities = np.zeros(2, dtype=float)

        for tree, weight in zip(self.trees, self.weights):
            probabilities += weight * self.predict_tree(tree, values)

        probabilities /= sum(self.weights)

        return {
            "prediction": int(np.argmax(probabilities)),
            "probability_no_detection": float(probabilities[0]),
            "probability_detection": float(probabilities[1]),
        }


if __name__ == "__main__":
    predictor = GeographicPredictor()

    result = predictor.predict(
        lat_grid=-33,
        lon_grid=151,
        month=12,
        previous_1=10,
        previous_2=8,
        previous_3=5,
    )

    print("\n=== GEOGRAPHIC PREDICTION TEST ===")
    print("Prediction:", result["prediction"])
    print(
        "Detection probability:",
        f"{result['probability_detection']:.2%}",
    )
    print(
        "No-detection probability:",
        f"{result['probability_no_detection']:.2%}",
    )
    print("\nPREDICTION_ENGINE_SUCCESS")
