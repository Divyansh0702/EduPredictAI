"""
Unit Tests for Model Prediction and Academic Support Recommendations
Uses small synthetic fixtures clearly distinguished from actual evaluation data.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import CLASS_NAMES
from src.predict import (
    is_model_trained,
    load_best_pipeline,
    load_evaluation_results,
    predict_bulk,
    predict_single,
)
from src.recommendations import generate_academic_recommendations


# Synthetic test fixture simulating an at-risk student record
SYNTHETIC_AT_RISK_STUDENT = {
    "school": "GP",
    "sex": "M",
    "age": 18,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 1,
    "Fedu": 1,
    "Mjob": "other",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 2,
    "studytime": 1,    # < 2 hours
    "failures": 2,     # Multiple past failures
    "schoolsup": "no",
    "famsup": "no",
    "paid": "no",
    "activities": "no",
    "nursery": "yes",
    "higher": "no",
    "internet": "yes",
    "romantic": "no",
    "famrel": 3,
    "freetime": 4,
    "goout": 4,
    "Dalc": 3,
    "Walc": 4,
    "health": 3,
    "absences": 14,    # High absences
    "G1": 6,          # Failing grade
    "G2": 7,          # Failing grade
}

# Synthetic test fixture simulating a high-performing student record
SYNTHETIC_HIGH_PERFORMING_STUDENT = {
    "school": "GP",
    "sex": "F",
    "age": 16,
    "address": "U",
    "famsize": "LE3",
    "Pstatus": "T",
    "Medu": 4,
    "Fedu": 4,
    "Mjob": "teacher",
    "Fjob": "services",
    "reason": "reputation",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 4,    # > 10 hours
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "yes",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 5,
    "freetime": 2,
    "goout": 2,
    "Dalc": 1,
    "Walc": 1,
    "health": 5,
    "absences": 2,
    "G1": 17,
    "G2": 18,
}


def test_model_trained_and_loadable():
    """Verify that serialized pipeline and evaluation artifacts exist and load correctly."""
    assert is_model_trained() is True
    pipeline = load_best_pipeline()
    assert pipeline is not None
    assert hasattr(pipeline, "predict")

    eval_results = load_evaluation_results()
    assert "models" in eval_results
    assert "metadata" in eval_results
    assert "winning_model" in eval_results["metadata"]


def test_predict_single_at_risk():
    """Verify single student inference on synthetic at-risk fixture."""
    pipeline = load_best_pipeline()
    result = predict_single(pipeline, SYNTHETIC_AT_RISK_STUDENT)

    assert "prediction" in result
    assert result["prediction"] in CLASS_NAMES
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)

    # Probabilities must sum close to 1.0
    if result["probabilities"]:
        total_prob = sum(result["probabilities"].values())
        assert abs(total_prob - 1.0) < 0.05

    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "risk_level" in result


def test_predict_single_high_performer():
    """Verify single student inference on synthetic high performing fixture."""
    pipeline = load_best_pipeline()
    result = predict_single(pipeline, SYNTHETIC_HIGH_PERFORMING_STUDENT)

    assert result["prediction"] in CLASS_NAMES
    assert "High" in result["prediction"] or "Medium" in result["prediction"]


def test_predict_bulk_synthetic():
    """Verify batch prediction on small synthetic DataFrame fixture."""
    pipeline = load_best_pipeline()
    synthetic_batch = pd.DataFrame([
        SYNTHETIC_AT_RISK_STUDENT,
        SYNTHETIC_HIGH_PERFORMING_STUDENT
    ])

    results_df, summary = predict_bulk(pipeline, synthetic_batch)

    assert len(results_df) == 2
    assert "Predicted_Performance" in results_df.columns
    assert summary["total_records"] == 2
    assert "category_distribution" in summary


def test_academic_recommendations_rules():
    """Verify that rule-based recommendation logic triggers expected support suggestions."""
    recs = generate_academic_recommendations(SYNTHETIC_AT_RISK_STUDENT, "Low (At-Risk)")

    assert "urgent_actions" in recs
    assert "recommended_actions" in recs
    assert "disclaimer" in recs
    assert len(recs["urgent_actions"]) > 0

    # Indicators for low study time, past failures, low scores, or absences must be triggered
    indicators = [a["indicator"] for a in recs["urgent_actions"]]
    assert any("Study Time" in ind or "Scores" in ind or "Grades" in ind or "Absences" in ind or "Failures" in ind for ind in indicators)

    # Mandatory educational disclaimer must be present
    assert "Educational Advisory Notice" in recs["disclaimer"]

