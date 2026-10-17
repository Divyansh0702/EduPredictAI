"""
Evaluation Module for EduPredict AI
Computes comprehensive classification metrics on held-out test data,
including Accuracy, Weighted F1, Macro F1, Precision, Recall, and Confusion Matrices.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_loader import CLASS_NAMES


def evaluate_classifier(
    model: Any,
    X_test: Any,
    y_test: Any,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate a fitted scikit-learn pipeline or classifier on test data.
    Computes accuracy, precision (weighted & macro), recall (weighted & macro),
    F1-score (weighted & macro), confusion matrix, and detailed per-class report.
    """
    labels = class_names or CLASS_NAMES

    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec_weighted = float(precision_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0
    )

    # Class distribution in ground truth test set vs predictions
    y_test_list = list(y_test)
    y_pred_list = list(y_pred)
    actual_counts = {lbl: y_test_list.count(lbl) for lbl in labels}
    predicted_counts = {lbl: y_pred_list.count(lbl) for lbl in labels}

    return {
        "accuracy": acc,
        "precision_weighted": prec_weighted,
        "precision_macro": prec_macro,
        "recall_weighted": rec_weighted,
        "recall_macro": rec_macro,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "class_names": labels,
        "classification_report": report_dict,
        "actual_distribution": actual_counts,
        "predicted_distribution": predicted_counts,
        "test_sample_size": len(y_test)
    }

