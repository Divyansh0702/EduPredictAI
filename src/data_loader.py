"""
Data Loader Module for EduPredict AI
Handles loading, auto-delimiter detection, validation, and target mapping
for the UCI Student Performance Dataset and compatible user uploads.
"""

import io
import os
from typing import Dict, Optional, Tuple, Union
import pandas as pd

# Default file locations within project according to academic subjects
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "artificial_intelligence_students.csv")
SECONDARY_DATA_PATH = os.path.join(DATA_DIR, "cloud_computing_students.csv")
FALLBACK_DATA_PATH = os.path.join(DATA_DIR, "student-mat.csv")
FALLBACK_SECONDARY_DATA_PATH = os.path.join(DATA_DIR, "student-por.csv")

# Target definition and thresholds
TARGET_RAW_COL = "G3"
TARGET_CATEGORY_COL = "performance_category"

# Categorization Thresholds for End Semester Exam (G3) (0 - 20 scale):
# - Low (At-Risk): End Semester Exam < 10 (Failing grade; requires academic intervention)
# - Medium (Satisfactory): 10 <= End Semester Exam <= 13 (Standard passing grade)
# - High (Good): 14 <= End Semester Exam <= 20 (Above-average to excellent standing)
THRESHOLDS = {
    "low_max": 9,        # 0 to 9 -> Low
    "medium_min": 10,    # 10 to 13 -> Medium
    "medium_max": 13,
    "high_min": 14,      # 14 to 20 -> High
    "high_max": 20
}

CLASS_NAMES = ["Low (At-Risk)", "Medium (Satisfactory)", "High (Good)"]

# Canonical expected columns from the UCI Student Performance dataset
EXPECTED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2", "G3"
]

CORE_FEATURE_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2"
]


def detect_delimiter(filepath_or_buffer: Union[str, io.StringIO, io.BytesIO]) -> str:
    """
    Detect whether the file uses a semicolon (UCI standard) or comma delimiter.
    """
    sample_text = ""
    if isinstance(filepath_or_buffer, str):
        if not os.path.exists(filepath_or_buffer):
            raise FileNotFoundError(f"Dataset file not found at: {filepath_or_buffer}")
        with open(filepath_or_buffer, "r", encoding="utf-8", errors="ignore") as f:
            sample_text = f.readline() + "\n" + f.readline()
    elif isinstance(filepath_or_buffer, io.BytesIO):
        pos = filepath_or_buffer.tell()
        sample_bytes = filepath_or_buffer.read(2048)
        filepath_or_buffer.seek(pos)
        sample_text = sample_bytes.decode("utf-8", errors="ignore")
    elif isinstance(filepath_or_buffer, io.StringIO):
        pos = filepath_or_buffer.tell()
        sample_text = filepath_or_buffer.read(2048)
        filepath_or_buffer.seek(pos)
    else:
        sample_text = ""

    # Count delimiters in header line
    first_line = sample_text.splitlines()[0] if sample_text else ""
    semicolon_count = first_line.count(";")
    comma_count = first_line.count(",")

    return ";" if semicolon_count > comma_count else ","


def load_dataset(
    filepath_or_buffer: Optional[Union[str, io.StringIO, io.BytesIO]] = None,
    sep: Optional[str] = None
) -> pd.DataFrame:
    """
    Load a student dataset from a file path or upload buffer.
    Automatically detects delimiters (comma vs semicolon) and strips quotes.
    """
    target_src = filepath_or_buffer or DEFAULT_DATA_PATH

    if isinstance(target_src, str) and not os.path.exists(target_src):
        if target_src == DEFAULT_DATA_PATH and os.path.exists(FALLBACK_DATA_PATH):
            target_src = FALLBACK_DATA_PATH
        elif target_src == SECONDARY_DATA_PATH and os.path.exists(FALLBACK_SECONDARY_DATA_PATH):
            target_src = FALLBACK_SECONDARY_DATA_PATH
        elif os.path.exists(DEFAULT_DATA_PATH):
            target_src = DEFAULT_DATA_PATH
        elif os.path.exists(FALLBACK_DATA_PATH):
            target_src = FALLBACK_DATA_PATH
        else:
            raise FileNotFoundError(
                f"Student performance dataset was not found at '{target_src}'. "
                "Please place artificial_intelligence_students.csv or student-mat.csv in the 'data/' folder."
            )

    detected_sep = sep if sep is not None else detect_delimiter(target_src)

    try:
        df = pd.read_csv(target_src, sep=detected_sep, quotechar='"')
    except Exception as e:
        # Fallback to python engine with auto-sniffing
        df = pd.read_csv(target_src, sep=None, engine="python")

    # Clean whitespace from column names and string values
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().str.strip('"')

    return df


def categorize_grade(grade: Union[int, float]) -> str:
    """
    Map numerical G3 grade (0 - 20) to educational performance category.
    """
    try:
        val = float(grade)
    except (ValueError, TypeError):
        return "Unknown"

    if val < THRESHOLDS["medium_min"]:
        return "Low (At-Risk)"
    elif val <= THRESHOLDS["medium_max"]:
        return "Medium (Satisfactory)"
    else:
        return "High (Good)"


def prepare_target(df: pd.DataFrame, target_col: str = TARGET_RAW_COL) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts and categorizes the target variable, returning feature dataframe X and target Series y.
    Ensures G3 is removed from X to prevent target leakage.
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Required target column '{target_col}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Clean target numeric values
    df_clean = df.copy()
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors="coerce")
    df_clean = df_clean.dropna(subset=[target_col])

    # Generate categorical labels
    y = df_clean[target_col].apply(categorize_grade)
    y.name = TARGET_CATEGORY_COL

    # Prevent target leakage: Drop G3 and target category from X
    cols_to_drop = [c for c in [target_col, TARGET_CATEGORY_COL] if c in df_clean.columns]
    X = df_clean.drop(columns=cols_to_drop)

    return X, y


def get_dataset_metadata(df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics and metadata for the loaded dataset.
    """
    has_g3 = TARGET_RAW_COL in df.columns
    total_records = len(df)
    avg_score = float(df[TARGET_RAW_COL].mean()) if has_g3 else None
    min_score = float(df[TARGET_RAW_COL].min()) if has_g3 else None
    max_score = float(df[TARGET_RAW_COL].max()) if has_g3 else None

    # Distribution of target if available
    target_dist = {}
    if has_g3:
        categories = df[TARGET_RAW_COL].apply(categorize_grade)
        counts = categories.value_counts().to_dict()
        for cat in CLASS_NAMES:
            target_dist[cat] = int(counts.get(cat, 0))

    return {
        "total_records": total_records,
        "has_target": has_g3,
        "average_score": avg_score,
        "min_score": min_score,
        "max_score": max_score,
        "target_distribution": target_dist,
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values_count": int(df.isnull().sum().sum()),
    }

