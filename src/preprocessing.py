"""
Preprocessing Module for EduPredict AI
Defines scikit-learn preprocessing pipelines, explicit feature lists,
input validation routines, and metadata for interactive forms.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Categorical features in the UCI Student Performance dataset
CATEGORICAL_FEATURES: List[str] = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]

# Numerical features in the UCI Student Performance dataset
NUMERICAL_FEATURES: List[str] = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
]

ALL_FEATURE_COLUMNS: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Feature specifications for validation and form rendering
FEATURE_METADATA: Dict[str, Dict] = {
    "age": {
        "label": "Student Age",
        "type": "numeric",
        "min": 15,
        "max": 22,
        "default": 17,
        "help": "Student age in years (15 to 22)",
    },
    "studytime": {
        "label": "Weekly Study Time",
        "type": "numeric",
        "min": 1,
        "max": 4,
        "default": 2,
        "options_map": {
            1: "< 2 hours",
            2: "2 to 5 hours",
            3: "5 to 10 hours",
            4: "> 10 hours",
        },
        "help": "Weekly study time category: 1 (<2h), 2 (2-5h), 3 (5-10h), 4 (>10h)",
    },
    "failures": {
        "label": "Past Class Failures",
        "type": "numeric",
        "min": 0,
        "max": 3,
        "default": 0,
        "options_map": {
            0: "0 failures",
            1: "1 failure",
            2: "2 failures",
            3: "3 or more failures",
        },
        "help": "Number of previous class failures (0 to 3)",
    },
    "G1": {
        "label": "First Period Grade (G1)",
        "type": "numeric",
        "min": 0,
        "max": 20,
        "default": 11,
        "help": "First trimester academic score (0 - 20 scale)",
    },
    "G2": {
        "label": "Second Period Grade (G2)",
        "type": "numeric",
        "min": 0,
        "max": 20,
        "default": 11,
        "help": "Second trimester academic score (0 - 20 scale)",
    },
    "absences": {
        "label": "School Absences",
        "type": "numeric",
        "min": 0,
        "max": 93,
        "default": 4,
        "help": "Total number of school absences",
    },
    "Medu": {
        "label": "Mother's Education Level",
        "type": "numeric",
        "min": 0,
        "max": 4,
        "default": 2,
        "options_map": {
            0: "None",
            1: "Primary Education (4th grade)",
            2: "5th to 9th grade",
            3: "Secondary Education",
            4: "Higher Education",
        },
        "help": "Mother's highest educational attainment",
    },
    "Fedu": {
        "label": "Father's Education Level",
        "type": "numeric",
        "min": 0,
        "max": 4,
        "default": 2,
        "options_map": {
            0: "None",
            1: "Primary Education (4th grade)",
            2: "5th to 9th grade",
            3: "Secondary Education",
            4: "Higher Education",
        },
        "help": "Father's highest educational attainment",
    },
    "traveltime": {
        "label": "Travel Time to School",
        "type": "numeric",
        "min": 1,
        "max": 4,
        "default": 1,
        "options_map": {
            1: "< 15 min",
            2: "15 to 30 min",
            3: "30 min to 1 hour",
            4: "> 1 hour",
        },
        "help": "Home to school transit duration",
    },
    "famrel": {
        "label": "Family Relationship Quality",
        "type": "numeric",
        "min": 1,
        "max": 5,
        "default": 4,
        "options_map": {
            1: "1 - Very Bad",
            2: "2 - Poor",
            3: "3 - Average",
            4: "4 - Good",
            5: "5 - Excellent",
        },
        "help": "Quality of family relationships (1 to 5)",
    },
    "freetime": {
        "label": "Free Time After School",
        "type": "numeric",
        "min": 1,
        "max": 5,
        "default": 3,
        "options_map": {
            1: "1 - Very Low",
            2: "2 - Low",
            3: "3 - Moderate",
            4: "4 - High",
            5: "5 - Very High",
        },
        "help": "Free time available after school (1 to 5)",
    },
    "goout": {
        "label": "Going Out with Friends",
        "type": "numeric",
        "min": 1,
        "max": 5,
        "default": 3,
        "options_map": {
            1: "1 - Very Low",
            2: "2 - Low",
            3: "3 - Moderate",
            4: "4 - High",
            5: "5 - Very High",
        },
        "help": "Frequency of going out with peers (1 to 5)",
    },
    "Dalc": {
        "label": "Workday Alcohol Consumption",
        "type": "numeric",
        "min": 1,
        "max": 5,
        "default": 1,
        "options_map": {
            1: "1 - Very Low",
            2: "2 - Low",
            3: "3 - Moderate",
            4: "4 - High",
            5: "5 - Very High",
        },
        "help": "Workday alcohol consumption scale (1 to 5)",
    },
    "Walc": {
        "label": "Weekend Alcohol Consumption",
        "type": "numeric",
        "min": 1,
        "max": 5,
        "default": 1,
        "options_map": {
            1: "1 - Very Low",
            2: "2 - Low",
            3: "3 - Moderate",
            4: "4 - High",
            5: "5 - Very High",
        },
        "help": "Weekend alcohol consumption scale (1 to 5)",
    },
    "health": {
        "label": "Current Health Status",
        "type": "numeric",
        "min": 1,
        "max": 5,
        "default": 4,
        "options_map": {
            1: "1 - Very Bad",
            2: "2 - Poor",
            3: "3 - Average",
            4: "4 - Good",
            5: "5 - Very Good",
        },
        "help": "Current physical health status (1 to 5)",
    },
    "school": {
        "label": "School",
        "type": "categorical",
        "options": ["GP", "MS"],
        "options_map": {
            "GP": "Gabriel Pereira",
            "MS": "Mousinho da Silveira",
        },
        "default": "GP",
        "help": "Student attending school",
    },
    "sex": {
        "label": "Sex",
        "type": "categorical",
        "options": ["F", "M"],
        "options_map": {"F": "Female", "M": "Male"},
        "default": "F",
        "help": "Student biological sex",
    },
    "address": {
        "label": "Address Type",
        "type": "categorical",
        "options": ["U", "R"],
        "options_map": {"U": "Urban", "R": "Rural"},
        "default": "U",
        "help": "Student home address environment",
    },
    "famsize": {
        "label": "Family Size",
        "type": "categorical",
        "options": ["GT3", "LE3"],
        "options_map": {"GT3": "Greater than 3", "LE3": "3 or fewer (<= 3)"},
        "default": "GT3",
        "help": "Family household size",
    },
    "Pstatus": {
        "label": "Parents' Cohabitation Status",
        "type": "categorical",
        "options": ["T", "A"],
        "options_map": {"T": "Living Together", "A": "Living Apart"},
        "default": "T",
        "help": "Parental living arrangement",
    },
    "Mjob": {
        "label": "Mother's Occupation",
        "type": "categorical",
        "options": ["at_home", "health", "other", "services", "teacher"],
        "options_map": {
            "at_home": "At Home",
            "health": "Health Care",
            "other": "Other",
            "services": "Civil Services",
            "teacher": "Teacher",
        },
        "default": "other",
        "help": "Mother's occupational category",
    },
    "Fjob": {
        "label": "Father's Occupation",
        "type": "categorical",
        "options": ["at_home", "health", "other", "services", "teacher"],
        "options_map": {
            "at_home": "At Home",
            "health": "Health Care",
            "other": "Other",
            "services": "Civil Services",
            "teacher": "Teacher",
        },
        "default": "other",
        "help": "Father's occupational category",
    },
    "reason": {
        "label": "Reason to Choose School",
        "type": "categorical",
        "options": ["course", "home", "other", "reputation"],
        "options_map": {
            "course": "Course Preference",
            "home": "Close to Home",
            "reputation": "School Reputation",
            "other": "Other",
        },
        "default": "course",
        "help": "Primary motivation for choosing the school",
    },
    "guardian": {
        "label": "Student Guardian",
        "type": "categorical",
        "options": ["mother", "father", "other"],
        "options_map": {
            "mother": "Mother",
            "father": "Father",
            "other": "Other Legal Guardian",
        },
        "default": "mother",
        "help": "Primary legal guardian",
    },
    "schoolsup": {
        "label": "Extra Educational School Support",
        "type": "categorical",
        "options": ["no", "yes"],
        "options_map": {"no": "No", "yes": "Yes"},
        "default": "no",
        "help": "Whether the school provides extra learning support",
    },
    "famsup": {
        "label": "Family Educational Support",
        "type": "categorical",
        "options": ["no", "yes"],
        "options_map": {"no": "No", "yes": "Yes"},
        "default": "yes",
        "help": "Whether family provides academic assistance and support",
    },
    "paid": {
        "label": "Extra Paid Classes",
        "type": "categorical",
        "options": ["no", "yes"],
        "options_map": {"no": "No", "yes": "Yes"},
        "default": "no",
        "help": "Private tutoring / extra paid coursework",
    },
    "activities": {
        "label": "Extra-Curricular Activities",
        "type": "categorical",
        "options": ["no", "yes"],
        "options_map": {"no": "No", "yes": "Yes"},
        "default": "no",
        "help": "Participation in school or club activities",
    },
    "nursery": {
        "label": "Attended Nursery School",
        "type": "categorical",
        "options": ["no", "yes"],
        "options_map": {"no": "No", "yes": "Yes"},
        "default": "yes",
        "help": "Prior early-childhood nursery education",
    },
    "higher": {
        "label": "Wants Higher Education",
        "type": "categorical",
        "options": ["no", "yes"],
        "options_map": {"no": "No", "yes": "Yes"},
        "default": "yes",
        "help": "Aspiration to attend university / higher education",
    },
    "internet": {
        "label": "Internet Access at Home",
        "type": "categorical",
        "options": ["no", "yes"],
        "options_map": {"no": "No", "yes": "Yes"},
        "default": "yes",
        "help": "Availability of home internet connectivity",
    },
    "romantic": {
        "label": "In a Romantic Relationship",
        "type": "categorical",
        "options": ["no", "yes"],
        "options_map": {"no": "No", "yes": "Yes"},
        "default": "no",
        "help": "Current romantic relationship status",
    },
}


def build_preprocessor() -> ColumnTransformer:
    """
    Construct scikit-learn ColumnTransformer for numerical and categorical features.
    - Numerical: Median imputation + Standard scaling.
    - Categorical: Most frequent imputation + OneHotEncoder with unknown handling.
    """
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def validate_input_features(
    df: pd.DataFrame, strict: bool = False
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate input DataFrame against required feature schema.
    Returns:
        (is_valid, list_of_errors, list_of_warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Check for missing required columns
    missing_cols = [c for c in ALL_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns ({len(missing_cols)}): {missing_cols}")

    # Check for unsupported/extra columns (subject column is recognized and accepted)
    extra_cols = [c for c in df.columns if c not in ALL_FEATURE_COLUMNS and c not in ["G3", "subject", "Subject"]]
    if extra_cols:
        warnings.append(f"Found {len(extra_cols)} extra columns that will be ignored: {extra_cols[:5]}")

    # If missing required columns, cannot proceed with value range check
    if errors:
        return False, errors, warnings

    # Range and type validation
    for num_col in NUMERICAL_FEATURES:
        meta = FEATURE_METADATA.get(num_col, {})
        col_series = pd.to_numeric(df[num_col], errors="coerce")
        null_count = int(col_series.isnull().sum())
        if null_count > 0:
            if strict:
                errors.append(f"Column '{num_col}' contains {null_count} non-numeric or missing entries.")
            else:
                warnings.append(f"Column '{num_col}' contains {null_count} missing values; will be imputed.")

        min_val = meta.get("min")
        max_val = meta.get("max")
        if min_val is not None and (col_series < min_val).any():
            warnings.append(
                f"Column '{num_col}' has values lower than expected minimum ({min_val})."
            )
        if max_val is not None and (col_series > max_val).any():
            warnings.append(
                f"Column '{num_col}' has values higher than expected maximum ({max_val})."
            )

    for cat_col in CATEGORICAL_FEATURES:
        meta = FEATURE_METADATA.get(cat_col, {})
        allowed = set(meta.get("options", []))
        str_series = df[cat_col].astype(str).str.strip().str.strip('"')
        unknown_vals = set(str_series.dropna().unique()) - allowed
        if unknown_vals:
            warnings.append(
                f"Column '{cat_col}' has unknown categories {unknown_vals}; encoder will handle gracefully."
            )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings

