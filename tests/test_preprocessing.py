"""
Unit Tests for Data Loading, Validation, and Preprocessing
"""

import io
import os
import sys
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    CLASS_NAMES,
    DEFAULT_DATA_PATH,
    categorize_grade,
    detect_delimiter,
    load_dataset,
    prepare_target,
)
from src.preprocessing import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
    validate_input_features,
)


def test_categorize_grade_thresholds():
    """Verify exact 0-30 grading thresholds."""
    # Low / At-Risk: < 15
    assert categorize_grade(0) == "Low (At-Risk)"
    assert categorize_grade(5) == "Low (At-Risk)"
    assert categorize_grade(14.9) == "Low (At-Risk)"

    # Medium / Satisfactory: 15 <= x <= 20
    assert categorize_grade(15) == "Medium (Satisfactory)"
    assert categorize_grade(18) == "Medium (Satisfactory)"
    assert categorize_grade(20) == "Medium (Satisfactory)"

    # High / Good: >= 21
    assert categorize_grade(21) == "High (Good)"
    assert categorize_grade(26) == "High (Good)"
    assert categorize_grade(30) == "High (Good)"


def test_detect_delimiter():
    """Verify delimiter auto-detection for semicolon and comma formats."""
    csv_comma = "school,sex,age,G3\nGP,F,17,12\n"
    csv_semi = "school;sex;age;G3\n\"GP\";\"F\";17;12\n"

    assert detect_delimiter(io.StringIO(csv_comma)) == ","
    assert detect_delimiter(io.StringIO(csv_semi)) == ";"


def test_load_dataset_default():
    """Verify loading of the default UCI student dataset."""
    df = load_dataset(DEFAULT_DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 100
    assert "G3" in df.columns
    assert "school" in df.columns
    assert "age" in df.columns


def test_prepare_target_prevents_leakage():
    """Verify that G3 is stripped from feature matrix X to eliminate target leakage."""
    df = load_dataset(DEFAULT_DATA_PATH)
    X, y = prepare_target(df)

    # G3 must NOT exist in X
    assert "G3" not in X.columns
    assert "performance_category" not in X.columns

    # y must have matching length and valid class names
    assert len(y) == len(X)
    assert set(y.unique()).issubset(set(CLASS_NAMES))


def test_build_preprocessor_pipeline_transformation():
    """Verify that scikit-learn ColumnTransformer processes numeric and categorical columns."""
    df = load_dataset(DEFAULT_DATA_PATH)
    X, _ = prepare_target(df)

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed is not None
    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] > len(ALL_FEATURE_COLUMNS)  # One-hot expansion


def test_validate_input_features_flags_missing():
    """Verify that validate_input_features catches missing required columns."""
    sample_df = pd.DataFrame([{
        "school": "GP",
        "sex": "F",
        "age": 17,
        # Intentionally missing other columns
    }])

    is_valid, errors, _ = validate_input_features(sample_df)
    assert is_valid is False
    assert len(errors) > 0
    assert "Missing required columns" in errors[0]

