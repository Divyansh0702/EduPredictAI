"""
EduPredict AI — Student Performance Prediction System
Main Application Entry Point and Executive Overview Dashboard.
"""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    CLASS_NAMES,
    DEFAULT_DATA_PATH,
    get_dataset_metadata,
    load_dataset,
)
from src.predict import is_model_trained, load_evaluation_results
from src.train_model import train_and_evaluate_all
from src.ui_components import (
    apply_custom_css,
    render_disclaimer,
    render_header,
    render_metric_card,
)

# Streamlit Page Config
st.set_page_config(
    page_title="EduPredict AI — Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

# Sidebar Navigation and System Status
st.sidebar.markdown("## 🎓 EduPredict AI")
st.sidebar.caption("Machine Learning & Applied AI Internship Project")
st.sidebar.markdown("---")

# Check model status
model_available = is_model_trained()
if model_available:
    st.sidebar.success("✅ Models Trained & Ready")
    try:
        eval_data = load_evaluation_results()
        winning_model = eval_data["metadata"]["winning_model"]
        st.sidebar.info(f"🏆 Active Model: **{winning_model}**")
    except Exception:
        pass
else:
    st.sidebar.warning("⚠️ Model Artifacts Missing")
    if st.sidebar.button("🚀 Train Models Now", use_container_width=True):
        with st.spinner("Training ML pipelines on UCI dataset..."):
            train_and_evaluate_all()
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Navigation")
st.sidebar.markdown(
    """
    - **Home Overview** (Current)
    - [1_Dashboard](Dashboard)
    - [2_Predict_Performance](Predict_Performance)
    - [3_Model_Analytics](Model_Analytics)
    - [4_Bulk_Prediction](Bulk_Prediction)
    """
)
st.sidebar.markdown("---")
st.sidebar.caption("Dataset: UCI Student Performance (Cortez & Silva, 2008)")

# Main Header
render_header(
    title="EduPredict AI — Student Performance Prediction System",
    subtitle=(
        "An end-to-end Applied Machine Learning application that forecasts student academic performance tiers "
        "and proactively identifies learners requiring targeted academic intervention and support."
    ),
    icon="🎓",
)

# Load dataset metadata for summary dashboard
try:
    df_raw = load_dataset()
    meta = get_dataset_metadata(df_raw)
except Exception as e:
    st.error(f"Error loading primary dataset: {e}")
    meta = {
        "total_records": 0,
        "average_score": None,
        "target_distribution": {},
        "missing_values_count": 0,
    }

# Load model evaluation results if available
eval_summary = {}
best_model_name = "Not Trained"
best_acc = 0.0
best_f1 = 0.0

if model_available:
    try:
        eval_payload = load_evaluation_results()
        best_model_name = eval_payload["metadata"]["winning_model"]
        best_metrics = eval_payload["models"][best_model_name]
        best_acc = best_metrics["accuracy"] * 100
        best_f1 = best_metrics["f1_weighted"]
    except Exception:
        pass

# Top Key Performance Indicators
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card(
        title="Student Records Analyzed",
        value=f"{meta['total_records']:,}",
        subtitle="Academic Cohort (AI & Cloud Computing)",
    )

with col2:
    avg_score_str = f"{meta['average_score']:.1f} / 20" if meta["average_score"] else "N/A"
    render_metric_card(
        title="Average End Semester Exam",
        value=avg_score_str,
        subtitle="End Semester Exam Mean Distribution",
    )

with col3:
    pass_pct = (
        f"{(meta['target_distribution'].get('Medium (Satisfactory)', 0) + meta['target_distribution'].get('High (Good)', 0)) / meta['total_records'] * 100:.1f}%"
        if meta["total_records"] > 0
        else "N/A"
    )
    render_metric_card(
        title="Overall Passing Rate",
        value=pass_pct,
        subtitle="Students passing End Semester Exam (>= 10 / 20)",
    )

with col4:
    acc_display = f"{best_acc:.1f}%" if model_available else "Pending"
    sub_text = f"Top Model: {best_model_name}" if model_available else "Run training script"
    render_metric_card(
        title="Test Accuracy (Winning Model)",
        value=acc_display,
        subtitle=sub_text,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Dashboard Charts & Architecture Section
chart_col, info_col = st.columns([1.1, 0.9])

with chart_col:
    st.markdown("### 📊 Distribution of Academic Performance Categories")
    if meta["target_distribution"]:
        dist_df = pd.DataFrame(
            [
                {"Category": cat, "Students": count}
                for cat, count in meta["target_distribution"].items()
            ]
        )

        colors_map = {
            "Low (At-Risk)": "#ef4444",
            "Medium (Satisfactory)": "#f59e0b",
            "High (Good)": "#10b981",
        }

        fig = px.pie(
            dist_df,
            names="Category",
            values="Students",
            color="Category",
            color_discrete_map=colors_map,
            hole=0.45,
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#ffffff", width=2)),
        )
        fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=340,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Performance category breakdown requires loaded dataset.")

with info_col:
    st.markdown("### 🎯 Core Objectives & Applied AI Workflow")
    st.markdown(
        """
        - **Proactive At-Risk Detection**: Identifies students prone to academic failure early, enabling timely educational guidance.
        - **Model Comparison**: Rigorously evaluates **Logistic Regression**, **Decision Tree**, and **Random Forest** algorithms on held-out test data.
        - **Strict Target Leakage Elimination**: Fully separates the End Semester Exam target ($G3$) from predictive features.
        - **Transparent Actionable Suggestions**: Delivers prioritized, rule-based academic support recommendations without opaque black-box claims.
        - **Enterprise Deployment**: Complete Streamlit application with single inference, bulk batch scoring, and downloadable CSV audit trails.
        """
    )
    if not model_available:
        st.warning("⚠️ Models have not been trained yet. Click below to initialize:")
        if st.button("🚀 Train Models Now (One-Click)", key="train_home"):
            with st.spinner("Training models..."):
                train_and_evaluate_all()
                st.success("Models trained successfully!")
                st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Quick Navigation Section
st.markdown("### 🚀 Application Modules")
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

with nav_col1:
    st.markdown(
        """
        <div class="metric-card" style="height: 190px;">
            <div class="metric-title">Module 1</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;">📈 Performance Dashboard</div>
            <div style="font-size: 0.85rem; color: #64748b; line-height: 1.4;">
                Deep-dive exploratory data analysis, correlation studies, and demographic visualizations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col2:
    st.markdown(
        """
        <div class="metric-card" style="height: 190px;">
            <div class="metric-title">Module 2</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;">🔮 Single Prediction</div>
            <div style="font-size: 0.85rem; color: #64748b; line-height: 1.4;">
                Interactive student profile assessment, risk probability distribution, and intervention suggestions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col3:
    st.markdown(
        """
        <div class="metric-card" style="height: 190px;">
            <div class="metric-title">Module 3</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;">🧠 Model Analytics</div>
            <div style="font-size: 0.85rem; color: #64748b; line-height: 1.4;">
                Comparative evaluation metrics, confusion matrix heatmaps, and feature importance rankings.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col4:
    st.markdown(
        """
        <div class="metric-card" style="height: 190px;">
            <div class="metric-title">Module 4</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;">📂 Bulk CSV Scoring</div>
            <div style="font-size: 0.85rem; color: #64748b; line-height: 1.4;">
                Batch student roster upload, schema validation, cohort risk summary, and CSV export.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer disclaimer
render_disclaimer()

