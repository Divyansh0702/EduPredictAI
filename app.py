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
    apply_plotly_styling,
    render_disclaimer,
    render_header,
    render_metric_card,
    render_sidebar_brand,
)

# Streamlit Page Config
st.set_page_config(
    page_title="EduPredict AI — Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

# Load model evaluation results if available
eval_summary = {}
best_model_name = "Not Trained"
best_acc = 0.0
best_f1 = 0.0
model_available = is_model_trained()

if model_available:
    try:
        eval_payload = load_evaluation_results()
        best_model_name = eval_payload["metadata"]["winning_model"]
        best_metrics = eval_payload["models"][best_model_name]
        best_acc = best_metrics["accuracy"] * 100
        best_f1 = best_metrics["f1_weighted"]
    except Exception:
        pass

# Sidebar Navigation and System Status
render_sidebar_brand(
    model_ready=model_available,
    winning_model=best_model_name if model_available else "Pending",
    accuracy_str=f"{best_acc:.1f}%" if model_available else "N/A"
)

if not model_available:
    st.sidebar.warning("⚠️ Model Artifacts Missing")
    if st.sidebar.button("🚀 Train Models Now", use_container_width=True):
        with st.spinner("Training ML pipelines on dataset..."):
            train_and_evaluate_all()
            st.rerun()

st.sidebar.markdown("### 📌 Navigation")
st.sidebar.markdown(
    """
    - **🏠 Home Overview** (Current)
    - [📈 1. Performance Dashboard](Dashboard)
    - [🔮 2. Predict Performance](Predict_Performance)
    - [🧠 3. Model Analytics](Model_Analytics)
    - [📂 4. Bulk CSV Scoring](Bulk_Prediction)
    """
)
st.sidebar.markdown("---")
st.sidebar.caption("Datasets: Artificial Intelligence & Cloud Computing Cohorts")

# Main Header
render_header(
    title="EduPredict AI — Academic Performance Prediction System",
    subtitle=(
        "An end-to-end Applied Machine Learning platform that forecasts student academic performance tiers "
        "and proactively identifies learners requiring targeted academic intervention and support."
    ),
    icon="🎓",
    badge="Applied Machine Learning & AI Internship"
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

# Top Key Performance Indicators
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card(
        title="Student Cohort Size",
        value=f"{meta['total_records']:,}",
        subtitle="Evaluated Course Roster",
        icon="👥",
        accent_color="#3b82f6"
    )

with col2:
    avg_score_str = f"{meta['average_score']:.1f} / 30" if meta["average_score"] else "N/A"
    render_metric_card(
        title="Mean End Semester Exam",
        value=avg_score_str,
        subtitle="Overall Average Score (0 - 30)",
        icon="📊",
        accent_color="#8b5cf6"
    )

with col3:
    pass_pct = (
        f"{(meta['target_distribution'].get('Medium (Satisfactory)', 0) + meta['target_distribution'].get('High (Good)', 0)) / meta['total_records'] * 100:.1f}%"
        if meta["total_records"] > 0
        else "N/A"
    )
    render_metric_card(
        title="Cohort Passing Rate",
        value=pass_pct,
        subtitle="Scoring >= 15 / 30 Passing Mark",
        icon="🎯",
        accent_color="#10b981"
    )

with col4:
    acc_display = f"{best_acc:.1f}%" if model_available else "Pending"
    sub_text = f"Champion: {best_model_name}" if model_available else "Run training script"
    render_metric_card(
        title="Model Test Accuracy",
        value=acc_display,
        subtitle=sub_text,
        icon="🏆",
        accent_color="#f59e0b"
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
            hole=0.52,
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#ffffff", width=2.5)),
            hovertemplate="<b>%{label}</b><br>Students: %{value}<br>Share: %{percent}<extra></extra>"
        )
        fig.update_layout(
            annotations=[dict(text=f"<b>{meta['total_records']}</b><br>Students", x=0.5, y=0.5, font_size=16, showarrow=False)],
            margin=dict(t=20, b=20, l=20, r=20),
            height=340,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Performance category breakdown requires loaded dataset.")

with info_col:
    st.markdown("### 🎯 Core Objectives & Applied AI Workflow")
    st.markdown(
        """
        <div class="glass-container" style="padding: 18px 22px;">
            <div style="font-size: 0.92rem; color: #334155; line-height: 1.65;">
                <p>• <strong>Proactive Early Warning:</strong> Identifies at-risk students before final exams using sessional indicators (ST-1, ST-2).</p>
                <p>• <strong>Comparative Algorithm Benchmarking:</strong> Rigorously tests <strong>Logistic Regression</strong>, <strong>Decision Trees</strong>, and <strong>Random Forests</strong> on held-out test data.</p>
                <p>• <strong>Zero Target Leakage:</strong> Strictly excludes the End Semester Exam grade ($G3$) from predictive features during training.</p>
                <p>• <strong>Transparent Counseling Heuristics:</strong> Generates tailored, rule-based academic support recommendations.</p>
                <p>• <strong>Batch Processing Engine:</strong> Automated CSV roster scoring with instant cohort risk breakdown and exportable reports.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
st.markdown("### 🚀 Interactive Application Modules")
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

with nav_col1:
    st.markdown(
        """
        <div class="metric-card" style="height: 195px;">
            <div class="metric-card-accent" style="background: linear-gradient(90deg, #3b82f6, #60a5fa);"></div>
            <div class="metric-title">Module 1</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;">📈 Performance Dashboard</div>
            <div style="font-size: 0.84rem; color: #64748b; line-height: 1.45;">
                Exploratory data analysis, score distributions across Mid-Terms, and study habit correlations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col2:
    st.markdown(
        """
        <div class="metric-card" style="height: 195px;">
            <div class="metric-card-accent" style="background: linear-gradient(90deg, #8b5cf6, #a78bfa);"></div>
            <div class="metric-title">Module 2</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;">🔮 Predict Performance</div>
            <div style="font-size: 0.84rem; color: #64748b; line-height: 1.45;">
                Live student evaluation form, End Semester forecast badge, probability chart, and intervention guide.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col3:
    st.markdown(
        """
        <div class="metric-card" style="height: 195px;">
            <div class="metric-card-accent" style="background: linear-gradient(90deg, #10b981, #34d399);"></div>
            <div class="metric-title">Module 3</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;">🧠 Model Analytics</div>
            <div style="font-size: 0.84rem; color: #64748b; line-height: 1.45;">
                Algorithm leaderboards, test accuracy, multi-metric bar charts, confusion matrices, and feature ranking.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col4:
    st.markdown(
        """
        <div class="metric-card" style="height: 195px;">
            <div class="metric-card-accent" style="background: linear-gradient(90deg, #f59e0b, #fbbf24);"></div>
            <div class="metric-title">Module 4</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;">📂 Bulk CSV Scoring</div>
            <div style="font-size: 0.84rem; color: #64748b; line-height: 1.45;">
                Batch student roster upload, schema validation, downloadable templates, and enriched CSV export.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer disclaimer
render_disclaimer()

