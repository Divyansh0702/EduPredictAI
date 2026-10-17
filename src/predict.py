"""
Prediction Module for EduPredict AI
Provides inference routines for single student prediction and bulk CSV scoring.
Handles input validation, probability estimation, confidence calculation,
and graceful error reporting.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import CLASS_NAMES
from src.preprocessing import ALL_FEATURE_COLUMNS, validate_input_features

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_PIPELINE_PATH = os.path.join(MODELS_DIR, "best_model_pipeline.joblib")
ALL_PIPELINES_PATH = os.path.join(MODELS_DIR, "all_pipelines.joblib")
RESULTS_PATH = os.path.join(MODELS_DIR, "evaluation_results.joblib")


def is_model_trained() -> bool:
    """Check if trained model pipeline and evaluation artifacts exist."""
    return os.path.exists(DEFAULT_PIPELINE_PATH) and os.path.exists(RESULTS_PATH)


def load_best_pipeline(pipeline_path: Optional[str] = None):
    """Load the best trained scikit-learn Pipeline."""
    path = pipeline_path or DEFAULT_PIPELINE_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trained model pipeline not found at '{path}'. "
            "Please run the model training script first (python src/train_model.py)."
        )
    return joblib.load(path)


def load_all_pipelines(pipelines_path: Optional[str] = None) -> Dict[str, Any]:
    """Load dictionary of all trained candidate pipelines."""
    path = pipelines_path or ALL_PIPELINES_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model pipelines archive not found at '{path}'.")
    return joblib.load(path)


def load_evaluation_results(results_path: Optional[str] = None) -> Dict[str, Any]:
    """Load evaluation metrics and metadata saved during training."""
    path = results_path or RESULTS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Evaluation results not found at '{path}'. "
            "Please run the model training script first."
        )
    return joblib.load(path)


def predict_single(
    pipeline: Any,
    student_features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate performance prediction and class probabilities for a single student.
    
    Args:
        pipeline: Fitted scikit-learn Pipeline.
        student_features: Dict containing values for all expected features.
        
    Returns:
        Dict containing prediction category, confidence score, probabilities map,
        and risk assessment.
    """
    # Create single-row DataFrame
    df_single = pd.DataFrame([student_features])

    # Validate schema
    is_valid, errors, warnings = validate_input_features(df_single, strict=False)
    if not is_valid:
        raise ValueError(f"Input validation error: {'; '.join(errors)}")

    # Ensure all required columns are present in consistent order
    df_input = df_single[ALL_FEATURE_COLUMNS].copy()

    # Predict class
    prediction = str(pipeline.predict(df_input)[0])

    # Probabilities
    probabilities: Dict[str, float] = {}
    confidence = 0.0

    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(df_input)[0]
            classes = pipeline.classes_
            for cls_name, prob in zip(classes, probs):
                probabilities[str(cls_name)] = round(float(prob), 4)
            confidence = round(float(max(probs)), 4)
        except Exception:
            probabilities = {}

    # Educational risk level tag
    if "Low" in prediction:
        risk_level = "High Risk (Immediate Academic Support Needed)"
        risk_color = "#dc2626"  # Red
    elif "Medium" in prediction:
        risk_level = "Moderate Risk (Targeted Reinforcement Recommended)"
        risk_color = "#f59e0b"  # Amber
    else:
        risk_level = "Low Risk (On Track / Strong Standing)"
        risk_color = "#10b981"  # Green

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "warnings": warnings,
        "input_features": student_features,
    }


def predict_bulk(
    pipeline: Any,
    input_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate batch predictions for multiple student records from an uploaded DataFrame.
    
    Args:
        pipeline: Fitted scikit-learn Pipeline.
        input_df: User-provided DataFrame.
        
    Returns:
        Tuple of:
            - Annotated DataFrame with predictions, confidence, and probability columns.
            - Summary metrics dictionary (total, valid, class distribution, error log).
    """
    df_copy = input_df.copy()

    # Clean whitespace from columns
    df_copy.columns = [str(c).strip() for c in df_copy.columns]

    # Validate columns
    is_valid, errors, warnings = validate_input_features(df_copy, strict=False)
    if not is_valid:
        raise ValueError(
            f"Uploaded file failed validation. {'; '.join(errors)}"
        )

    # Reorder columns to match pipeline training order
    X_input = df_copy[ALL_FEATURE_COLUMNS].copy()

    # Predict classes
    predictions = pipeline.predict(X_input)
    df_result = input_df.copy()
    df_result["Predicted_Performance"] = predictions

    # Predict probabilities if supported
    confidence_list = []
    if hasattr(pipeline, "predict_proba"):
        try:
            probs_matrix = pipeline.predict_proba(X_input)
            classes = pipeline.classes_
            for idx, cls_name in enumerate(classes):
                col_name = f"Probability_{cls_name.split()[0]}"
                df_result[col_name] = [round(float(p), 3) for p in probs_matrix[:, idx]]
            confidence_list = [round(float(max(row)), 3) for row in probs_matrix]
            df_result["Confidence"] = confidence_list
        except Exception:
            pass

    # Summary distribution
    counts = pd.Series(predictions).value_counts().to_dict()
    category_summary = {cls_name: int(counts.get(cls_name, 0)) for cls_name in CLASS_NAMES}

    at_risk_count = category_summary.get("Low (At-Risk)", 0)
    total_count = len(predictions)
    at_risk_pct = round((at_risk_count / total_count) * 100, 1) if total_count > 0 else 0.0

    summary = {
        "total_records": total_count,
        "category_distribution": category_summary,
        "at_risk_count": at_risk_count,
        "at_risk_percentage": at_risk_pct,
        "warnings": warnings,
    }

    return df_result, summary

