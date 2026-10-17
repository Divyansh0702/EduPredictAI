"""
Page 4: Bulk CSV Prediction & Cohort Risk Scoring
Enables batch scoring of student rosters via CSV upload, automated schema validation,
risk categorization, cohort distribution analysis, and CSV export.
"""

import io
import os
import sys
from typing import Optional
import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    CLASS_NAMES,
    DEFAULT_DATA_PATH,
    SECONDARY_DATA_PATH,
    detect_delimiter,
    load_dataset,
)
from src.predict import is_model_trained, load_best_pipeline, predict_bulk
from src.preprocessing import ALL_FEATURE_COLUMNS, validate_input_features
from src.train_model import train_and_evaluate_all
from src.ui_components import (
    apply_custom_css,
    render_disclaimer,
    render_header,
    render_metric_card,
)

st.set_page_config(
    page_title="Bulk Prediction — EduPredict AI",
    page_icon="📂",
    layout="wide",
)

apply_custom_css()

render_header(
    title="Bulk Student Cohort Prediction & Risk Scoring",
    subtitle=(
        "Upload a student roster CSV to generate batch performance predictions, identify at-risk clusters, "
        "and download enriched results with confidence probabilities."
    ),
    icon="📂",
)

# Model check
if not is_model_trained():
    st.warning("⚠️ Trained model pipeline not found on disk.")
    st.info("Train the model first to enable bulk scoring:")
    if st.button("🚀 Train Models Now", key="train_bulk"):
        with st.spinner("Training models..."):
            train_and_evaluate_all()
            st.success("Trained successfully!")
            st.rerun()
    st.stop()

try:
    pipeline = load_best_pipeline()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Template generator helper
def get_sample_template_csv(csv_path: Optional[str] = None) -> bytes:
    """Generate sample CSV template containing first 10 records from dataset."""
    target_path = csv_path or DEFAULT_DATA_PATH
    try:
        sample_df = load_dataset(target_path).head(10)
        # Drop G3 if present so template is pure feature input
        if "G3" in sample_df.columns:
            sample_df = sample_df.drop(columns=["G3"])
        return sample_df.to_csv(index=False).encode("utf-8")
    except Exception:
        # Fallback minimal template
        minimal_df = pd.DataFrame([{col: 0 if col in ["age", "G1", "G2"] else "no" for col in ALL_FEATURE_COLUMNS}])
        return minimal_df.to_csv(index=False).encode("utf-8")

# Template Download & Instructions
st.markdown("### 📥 1. Prepare Student Data")
inst_col1, inst_col2 = st.columns([1.1, 0.9])

with inst_col1:
    st.markdown(
        """
        Your CSV file should contain student attributes (study habits, mid-term assessment scores `G1`, `G2`, absences, etc.).
        - Delimiters supported: Comma (`,`) or Semicolon (`;`).
        - The `subject` column is optional and will be preserved in output.
        - You can download pre-populated templates for **Artificial Intelligence** or **Cloud Computing** below:
        """
    )

with inst_col2:
    ai_template = get_sample_template_csv(DEFAULT_DATA_PATH)
    cc_template = get_sample_template_csv(SECONDARY_DATA_PATH)
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.download_button(
            label="📄 Download AI Template",
            data=ai_template,
            file_name="artificial_intelligence_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d_col2:
        st.download_button(
            label="📄 Download Cloud Template",
            data=cc_template,
            file_name="cloud_computing_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown("---")

# File Upload Section
st.markdown("### 📤 2. Upload Student Roster CSV")
uploaded_file = st.file_uploader(
    "Choose a CSV file containing student records:",
    type=["csv"],
    help="Upload CSV with student features",
)

if uploaded_file is not None:
    try:
        # Detect delimiter and load
        detected_sep = detect_delimiter(uploaded_file)
        uploaded_df = pd.read_csv(uploaded_file, sep=detected_sep, quotechar='"')
        uploaded_df.columns = [c.strip() for c in uploaded_df.columns]

        st.success(f"Successfully read CSV with {len(uploaded_df)} student records and {len(uploaded_df.columns)} columns.")

        # Validate columns
        is_valid, errors, warnings = validate_input_features(uploaded_df, strict=False)

        if not is_valid:
            st.error("❌ Schema Validation Failed:")
            for err in errors:
                st.markdown(f"- **{err}**")
            st.info("Tip: Click 'Download Sample CSV Template' above to see the required column structure.")
        else:
            if warnings:
                with st.expander("⚠️ Schema Warnings / Auto-Handled Discrepancies"):
                    for warn in warnings:
                        st.markdown(f"- {warn}")

            with st.spinner("Executing batch inference across all records..."):
                results_df, summary = predict_bulk(pipeline, uploaded_df)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("## 📊 Cohort Prediction Summary")

            # KPI Cards
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            with kpi1:
                render_metric_card("Total Scored", f"{summary['total_records']:,}", "Valid records evaluated")

            with kpi2:
                render_metric_card(
                    "At-Risk Students",
                    f"{summary['at_risk_count']:,}",
                    f"{summary['at_risk_percentage']}% of cohort",
                )

            with kpi3:
                sat_count = summary["category_distribution"].get("Medium (Satisfactory)", 0)
                render_metric_card("Satisfactory Standing", f"{sat_count:,}", "Meeting baseline")

            with kpi4:
                good_count = summary["category_distribution"].get("High (Good)", 0)
                render_metric_card("High Standing", f"{good_count:,}", "Strong academic pace")

            # Cohort distribution visualization
            st.markdown("<br>", unsafe_allow_html=True)
            ch_col1, ch_col2 = st.columns([1, 1.2])

            with ch_col1:
                cat_chart_df = pd.DataFrame([
                    {"Category": cat, "Students": count}
                    for cat, count in summary["category_distribution"].items()
                ])
                fig_pie = px.pie(
                    cat_chart_df,
                    names="Category",
                    values="Students",
                    color="Category",
                    color_discrete_map={
                        "Low (At-Risk)": "#ef4444",
                        "Medium (Satisfactory)": "#f59e0b",
                        "High (Good)": "#10b981",
                    },
                    hole=0.4,
                    title="Predicted Cohort Risk Distribution",
                )
                fig_pie.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)

            with ch_col2:
                st.markdown("#### 🎯 Cohort Intervention Overview")
                if summary["at_risk_count"] > 0:
                    st.warning(
                        f"🚨 **Action Required:** {summary['at_risk_count']} student(s) ({summary['at_risk_percentage']}%) "
                        "have been flagged in the **Low (At-Risk)** category. Immediate targeted tutoring, attendance review, "
                        "and academic counseling are strongly recommended."
                    )
                else:
                    st.success("✅ Excellent cohort standing! Zero students were classified in the At-Risk tier.")

            # Filter and Display Results Table
            st.markdown("### 📋 Student Predictions Table")

            filter_choice = st.selectbox(
                "Filter View by Predicted Category:",
                options=["All Categories", "Low (At-Risk)", "Medium (Satisfactory)", "High (Good)"],
                index=0,
            )

            display_df = results_df.copy()
            if filter_choice != "All Categories":
                display_df = display_df[display_df["Predicted_Performance"] == filter_choice]

            # Reorder columns to put subject and prediction first for clear visibility
            priority_cols = ["subject", "Subject", "Predicted_Performance", "Confidence", "Probability_Low", "Probability_Medium", "Probability_High"]
            pred_cols = [c for c in priority_cols if c in display_df.columns]
            other_cols = [c for c in display_df.columns if c not in pred_cols]
            ordered_cols = pred_cols + other_cols

            st.dataframe(
                display_df[ordered_cols],
                use_container_width=True,
                height=400,
            )

            # CSV Download Button
            csv_output = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Complete Predictions as CSV",
                data=csv_output,
                file_name="edupredict_predictions_output.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )

    except Exception as e:
        st.error(f"Error processing uploaded CSV: {e}")

else:
    st.info("👆 Upload a CSV file or download the template above to run bulk predictions.")

render_disclaimer()

