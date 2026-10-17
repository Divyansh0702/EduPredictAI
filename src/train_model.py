"""
Model Training Module for EduPredict AI
Trains, evaluates, and compares three classification algorithms:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

Features:
- Reproducible train/test split with fixed random seed (42) and stratified sampling.
- Strict prevention of target leakage by isolating the target variable G3.
- End-to-end scikit-learn Pipelines combining ColumnTransformer and classifier.
- Selection of the best model based on weighted F1-score on held-out test data.
- Joblib serialization of pipelines, models, and evaluation results.
"""

import datetime
import json
import os
import sys
from typing import Any, Dict, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import (
    CLASS_NAMES,
    DEFAULT_DATA_PATH,
    get_dataset_metadata,
    load_dataset,
    prepare_target,
)
from src.evaluate import evaluate_classifier
from src.preprocessing import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def get_feature_names(preprocessor) -> list:
    """Extract output feature names from fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction
        names = list(NUMERICAL_FEATURES)
        try:
            cat_step = preprocessor.named_transformers_["cat"]
            encoder = cat_step.named_steps["onehot"]
            cat_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
            names.extend(list(cat_names))
        except Exception:
            pass
        return names


def extract_feature_importance(pipeline, model_name: str) -> Dict[str, float]:
    """
    Extract feature importances or absolute coefficient magnitudes from a fitted pipeline.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names(preprocessor)

    importances = {}
    if hasattr(classifier, "feature_importances_"):
        raw_imp = classifier.feature_importances_
        for name, imp in zip(feature_names, raw_imp):
            importances[name] = float(imp)
    elif hasattr(classifier, "coef_"):
        # Mean absolute coefficient across all target classes for multinomial logistic regression
        coefs = np.mean(np.abs(classifier.coef_), axis=0)
        for name, c in zip(feature_names, coefs):
            importances[name] = float(c)

    # Sort descending
    sorted_imp = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    return sorted_imp


def train_and_evaluate_all(
    data_path: str = DEFAULT_DATA_PATH,
    test_size: float = 0.20,
    random_seed: int = 42
) -> Tuple[Dict[str, Any], str, Dict[str, Pipeline]]:
    """
    Train and evaluate Logistic Regression, Decision Tree, and Random Forest.
    Returns:
        evaluation_results: dictionary containing metrics for each model
        best_model_name: name of the top-performing model
        pipelines: dictionary of fitted Pipeline objects
    """
    print(f"Loading dataset from: {data_path}")
    raw_df = load_dataset(data_path)
    dataset_summary = get_dataset_metadata(raw_df)

    # Extract target and features
    X, y = prepare_target(raw_df)
    print(f"Target distribution across classes:\n{y.value_counts()}")

    # Stratified split to ensure balanced class distribution across train/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y
    )
    print(f"Dataset split: {len(X_train)} training records, {len(X_test)} testing records.")

    # Model definitions
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=random_seed
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=6,
            min_samples_leaf=3,
            random_state=random_seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_seed
        )
    }

    results = {}
    fitted_pipelines = {}
    feature_importances = {}

    for name, clf in candidate_models.items():
        print(f"Training {name}...")
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Evaluate on held-out test data
        eval_metrics = evaluate_classifier(pipeline, X_test, y_test, CLASS_NAMES)
        results[name] = eval_metrics
        print(
            f"-> {name} Results: "
            f"Accuracy={eval_metrics['accuracy']:.4f}, "
            f"Weighted F1={eval_metrics['f1_weighted']:.4f}, "
            f"Macro F1={eval_metrics['f1_macro']:.4f}"
        )

        # Feature importances
        feature_importances[name] = extract_feature_importance(pipeline, name)

    # Model Selection Criterion: Highest Weighted F1-score on held-out test set
    best_model_name = max(results.keys(), key=lambda m: results[m]["f1_weighted"])
    print(f"\nWinning Model Selected: '{best_model_name}' (Highest Weighted F1: {results[best_model_name]['f1_weighted']:.4f})")

    # Limitations documentation
    limitations = {
        "dataset_bias": (
            "The model is trained on Portuguese secondary school students (Cortez & Silva, 2008). "
            "Socio-demographic patterns may differ across distinct educational cultures."
        ),
        "sample_size": (
            f"The training corpus comprises {len(raw_df)} student records. Rare academic trajectories "
            "or edge cases may have limited representation."
        ),
        "feature_availability": (
            "Prediction relies heavily on prior period grades (G1, G2), study habits, and absences. "
            "Omitting prior grades may reduce predictive power."
        ),
        "non_deterministic_human_factors": (
            "Predictions are statistical approximations based on historical data. They must NOT be treated "
            "as deterministic guarantees of student potential or ability."
        )
    }

    output_payload = {
        "metadata": {
            "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_path": data_path,
            "total_records": dataset_summary["total_records"],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_split_ratio": test_size,
            "random_seed": random_seed,
            "selection_criterion": "Highest Weighted F1-Score on held-out test set",
            "winning_model": best_model_name,
            "target_classes": CLASS_NAMES,
            "limitations": limitations,
            "dataset_summary": dataset_summary,
        },
        "models": results,
        "feature_importances": feature_importances,
    }

    # Save artifacts to models/ directory
    os.makedirs(MODELS_DIR, exist_ok=True)

    best_pipeline_path = os.path.join(MODELS_DIR, "best_model_pipeline.joblib")
    all_pipelines_path = os.path.join(MODELS_DIR, "all_pipelines.joblib")
    results_joblib_path = os.path.join(MODELS_DIR, "evaluation_results.joblib")
    results_json_path = os.path.join(MODELS_DIR, "evaluation_results.json")

    joblib.dump(fitted_pipelines[best_model_name], best_pipeline_path)
    joblib.dump(fitted_pipelines, all_pipelines_path)
    joblib.dump(output_payload, results_joblib_path)

    # Save JSON representation (converting non-serializable objects if any)
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Model artifacts successfully serialized to: {MODELS_DIR}")
    return output_payload, best_model_name, fitted_pipelines


if __name__ == "__main__":
    train_and_evaluate_all()
