"""Model training and benchmarking for ReconX.

This project intentionally uses a synthetic, lab-style dataset for educational use.
The dataset is larger and more structured than the starter version and includes:
- evidence metadata
- site-level grouping to reduce leakage
- multiple model comparisons with standard evaluation metrics
- persisted benchmark data for retraining and monitoring
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

DATA = Path(__file__).with_name("training_data.csv")
BENCHMARK_PATH = Path(__file__).with_name("model_benchmark.json")

FEATURES = [
    "https_missing",
    "missing_headers",
    "server_error",
    "resolved_ip_count",
    "certificate_invalid",
    "dns_invalid",
    "hsts_missing",
    "server_header_present",
]

MODEL_DEFINITIONS = {
    "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced_subsample",
    ),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}


def generate_documented_dataset(rows_per_site: int = 8, site_count: int = 500):
    """Create a larger dataset with evidence metadata and site grouping.

    This simulates documented lab examples rather than real production telemetry.
    """
    rng = random.Random(42)
    rows = []
    for site_idx in range(1, site_count + 1):
        site_id = f"site-{site_idx:04d}"
        source = "authorized-lab-example"
        evidence = (
            "TLS check, DNS resolution, status code, header review, and documented lab evidence. "
            "Not production-grade intelligence."
        )
        for sample_idx in range(rows_per_site):
            https_missing = int(rng.random() < 0.5)
            missing_headers = rng.randint(0, 4)
            server_error = int(rng.random() < 0.3)
            resolved_ip_count = rng.randint(1, 6)
            certificate_invalid = int(rng.random() < 0.35)
            dns_invalid = int(rng.random() < 0.2)
            hsts_missing = int(rng.random() < 0.55)
            server_header_present = int(rng.random() < 0.7)

            weighted_score = (
                28 * https_missing
                + 16 * missing_headers
                + 25 * server_error
                + 5 * max(0, resolved_ip_count - 1)
                + 18 * certificate_invalid
                + 22 * dns_invalid
                + 12 * hsts_missing
                + 8 * server_header_present
            )
            if weighted_score < 35:
                risk = "LOW"
                severity = "Info"
            elif weighted_score < 75:
                risk = "MEDIUM"
                severity = "Moderate"
            else:
                risk = "HIGH"
                severity = "High"

            rows.append(
                {
                    "site_id": site_id,
                    "source": source,
                    "evidence": evidence,
                    "severity": severity,
                    "https_missing": https_missing,
                    "missing_headers": missing_headers,
                    "server_error": server_error,
                    "resolved_ip_count": resolved_ip_count,
                    "certificate_invalid": certificate_invalid,
                    "dns_invalid": dns_invalid,
                    "hsts_missing": hsts_missing,
                    "server_header_present": server_header_present,
                    "risk": risk,
                }
            )
    return rows


def save_dataset(rows):
    fieldnames = [
        "site_id",
        "source",
        "evidence",
        "severity",
        *FEATURES,
        "risk",
    ]
    with DATA.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_dataset():
    if not DATA.exists() or DATA.stat().st_size < 200:
        rows = generate_documented_dataset()
        save_dataset(rows)

    with DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        rows = generate_documented_dataset()
        save_dataset(rows)

    X = [[int(row[column]) for column in FEATURES] for row in rows]
    y = [row["risk"] for row in rows]
    groups = [row["site_id"] for row in rows]
    return X, y, groups


def _site_group_split(X, y, groups):
    splitter = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return train_idx.tolist(), test_idx.tolist()


def _evaluate_classifier(model, X_test, y_test):
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["LOW", "MEDIUM", "HIGH"]).tolist(),
        "classification_report": classification_report(y_test, predictions, labels=["LOW", "MEDIUM", "HIGH"], zero_division=0),
    }
    false_negatives = confusion_matrix(y_test, predictions, labels=["LOW", "MEDIUM", "HIGH"])[1:, 0].sum()
    metrics["false_negative_rate"] = false_negatives / max(1, sum(1 for label in y_test if label in {"MEDIUM", "HIGH"}))
    return metrics


def compare_models():
    X, y, groups = load_dataset()
    train_idx, test_idx = _site_group_split(X, y, groups)
    X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
    y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

    model_results = {}
    for name, estimator in MODEL_DEFINITIONS.items():
        model = estimator
        model.fit(X_train, y_train)
        model_results[name] = _evaluate_classifier(model, X_test, y_test)

    best_model_name = max(
        model_results,
        key=lambda name: model_results[name]["f1_macro"],
    )
    benchmark = {
        "best_model": best_model_name,
        "models": model_results,
        "dataset": {
            "rows": len(y),
            "features": FEATURES,
            "sites": len(set(groups)),
        },
    }
    with BENCHMARK_PATH.open("w", encoding="utf-8") as handle:
        json.dump(benchmark, handle, indent=2)

    return benchmark


def train_or_load_model():
    X, y, groups = load_dataset()
    train_idx, test_idx = _site_group_split(X, y, groups)
    X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
    y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

    benchmark = compare_models()
    best_model_name = benchmark["best_model"]
    model = MODEL_DEFINITIONS[best_model_name]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, {
        "accuracy": accuracy,
        "best_model": best_model_name,
        "benchmark": benchmark,
    }


def predict_risk(model, features):
    prediction = model.predict([features])[0]
    return str(prediction).upper()


def get_model_benchmark():
    if BENCHMARK_PATH.exists():
        with BENCHMARK_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return compare_models()
