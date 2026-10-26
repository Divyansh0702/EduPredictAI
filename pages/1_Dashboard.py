"""
Page 1: Performance & Exploratory Analytics Dashboard
Interactive visualization of student demographics, study habits, grade trends,
and correlation patterns.
"""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    DEFAULT_DATA_PATH,
    SECONDARY_DATA_PATH,
    categorize_grade,
    load_dataset,
)
from src.predict import is_model_trained, load_evaluation_results
from src.ui_components import (
    apply_custom_css,
    apply_plotly_styling,
    render_disclaimer,
    render_header,
    render_metric_card,
    render_sidebar_brand,
)

st.set_page_config(
    page_title="Performance Dashboard — EduPredict AI",
    page_icon="📈",
    layout="wide",
)

apply_custom_css()

# Model status check for sidebar
model_available = is_model_trained()
winning_model = "Logistic Regression"
test_acc_str = "87.3%"
if model_available:
    try:
        eval_payload = load_evaluation_results()
        winning_model = eval_payload["metadata"]["winning_model"]
        acc = eval_payload["models"][winning_model]["accuracy"] * 100
        test_acc_str = f"{acc:.1f}%"
    except Exception:
        pass

render_sidebar_brand(model_ready=model_available, winning_model=winning_model, accuracy_str=test_acc_str)

render_header(
    title="Student Performance & Demographic Analytics",
    subtitle=(
        "Explore student cohort statistics, study habits, grade trajectories, and factors "
        "influencing academic outcomes in the curriculum."
    ),
    icon="📈",
    badge="Module 1: Exploratory Cohort Intelligence"
)

# Sidebar: Dataset selection & filters
st.sidebar.markdown("### 🗂️ Subject Selection")
dataset_choice = st.sidebar.radio(
    "Select Course Subject:",
    options=[
        "1. Artificial Intelligence",
        "2. Cloud Computing",
        "Upload Custom CSV"
    ],
    index=0,
)

df = None
selected_path = DEFAULT_DATA_PATH

if "Artificial Intelligence" in dataset_choice:
    selected_path = DEFAULT_DATA_PATH
    df = load_dataset(selected_path)
elif "Cloud Computing" in dataset_choice:
    if os.path.exists(SECONDARY_DATA_PATH):
        selected_path = SECONDARY_DATA_PATH
        df = load_dataset(selected_path)
    else:
        st.warning("Cloud Computing dataset not found in data/. Loading Artificial Intelligence dataset.")
        df = load_dataset(DEFAULT_DATA_PATH)
else:
    uploaded = st.sidebar.file_uploader("Upload student CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = load_dataset(uploaded)
            st.sidebar.success(f"Loaded {len(df)} records from uploaded file.")
        except Exception as e:
            st.sidebar.error(f"Failed to parse file: {e}")
            df = load_dataset(DEFAULT_DATA_PATH)
    else:
        st.sidebar.info("Using default Artificial Intelligence dataset until a file is uploaded.")
        df = load_dataset(DEFAULT_DATA_PATH)

# Add performance category column if G3 exists
has_g3 = "G3" in df.columns
if has_g3:
    df["Performance_Category"] = df["G3"].apply(categorize_grade)

# High-Level Metrics
col1, col2, col3, col4, col5 = st.columns(5)

total_students = len(df)
avg_g3 = float(df["G3"].mean()) if has_g3 else 0.0
median_absences = float(df["absences"].median()) if "absences" in df.columns else 0.0

pass_count = int((df["G3"] >= 15).sum()) if has_g3 else 0
pass_pct = (pass_count / total_students) * 100 if total_students > 0 else 0.0

good_count = int((df["G3"] >= 21).sum()) if has_g3 else 0
good_pct = (good_count / total_students) * 100 if total_students > 0 else 0.0

with col1:
    render_metric_card("Enrolled Students", f"{total_students:,}", "Active Student Cohort", icon="👥", accent_color="#3b82f6")
with col2:
    render_metric_card("Mean End Semester Exam", f"{avg_g3:.2f} / 30" if has_g3 else "N/A", "Scale: 0 to 30", icon="📊", accent_color="#8b5cf6")
with col3:
    render_metric_card("Passing Rate (>=15)", f"{pass_pct:.1f}%", f"{pass_count} Passed", icon="🎯", accent_color="#10b981")
with col4:
    render_metric_card("High Standing (>=21)", f"{good_pct:.1f}%", f"{good_count} Honors", icon="⭐", accent_color="#f59e0b")
with col5:
    render_metric_card("Median Absences", f"{median_absences:.0f} days", "Course attendance", icon="📅", accent_color="#06b6d4")

st.markdown("<br>", unsafe_allow_html=True)

# Tabs for Organized Analysis
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exam Distribution & Progression",
    "⏱️ Study Habits & Academic Risk",
    "🏡 Demographics & Support Systems",
    "🔍 Raw Data Explorer"
])

with tab1:
    st.markdown("#### Examination Grade Distributions & Progression")
    g_col1, g_col2 = st.columns([1, 1])

    with g_col1:
        if has_g3 and "G1" in df.columns and "G2" in df.columns:
            grades_melted = df.melt(
                value_vars=["G1", "G2", "G3"],
                var_name="Period",
                value_name="Score"
            )
            exam_label_map = {
                "G1": "ST - 1",
                "G2": "ST - 2",
                "G3": "End Semester Exam"
            }
            grades_melted["Assessment"] = grades_melted["Period"].map(exam_label_map)
            fig_box = px.box(
                grades_melted,
                x="Assessment",
                y="Score",
                color="Assessment",
                points="outliers",
                color_discrete_sequence=["#3b82f6", "#8b5cf6", "#10b981"],
                title="Progression: ST - 1, ST - 2 & End Semester Exam",
                labels={"Score": "Exam Score (0 - 30)", "Assessment": "Examination"}
            )
            apply_plotly_styling(fig_box, height=390)
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

    with g_col2:
        if has_g3:
            fig_hist = px.histogram(
                df,
                x="G3",
                nbins=31,
                color="Performance_Category",
                color_discrete_map={
                    "Low (At-Risk)": "#ef4444",
                    "Medium (Satisfactory)": "#f59e0b",
                    "High (Good)": "#10b981",
                },
                title="End Semester Exam Score Histogram with Performance Tiers",
                labels={"G3": "End Semester Exam Score (0 - 30)"}
            )
            fig_hist.add_vline(x=15, line_dash="dash", line_color="#ef4444", annotation_text="Passing Threshold (15)")
            fig_hist.add_vline(x=21, line_dash="dash", line_color="#10b981", annotation_text="High Standing (21)")
            apply_plotly_styling(fig_hist, height=390)
            st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.markdown("#### Study Habits, Prior Failures & Attendance")
    s_col1, s_col2 = st.columns(2)

    with s_col1:
        if "studytime" in df.columns and has_g3:
            study_labels = {1: "< 2 hrs/wk", 2: "2 - 5 hrs/wk", 3: "5 - 10 hrs/wk", 4: "> 10 hrs/wk"}
            df["Study_Time_Label"] = df["studytime"].map(study_labels)

            fig_study = px.box(
                df,
                x="Study_Time_Label",
                y="G3",
                color="Study_Time_Label",
                category_orders={"Study_Time_Label": ["< 2 hrs/wk", "2 - 5 hrs/wk", "5 - 10 hrs/wk", "> 10 hrs/wk"]},
                title="Impact of Weekly Study Time on End Semester Exam Score",
                labels={"G3": "End Semester Exam Score (0 - 30)"}
            )
            apply_plotly_styling(fig_study, height=380)
            fig_study.update_layout(showlegend=False)
            st.plotly_chart(fig_study, use_container_width=True)

    with s_col2:
        if "failures" in df.columns and has_g3:
            df["Failures_Label"] = df["failures"].astype(str) + " past failures"
            fig_fail = px.box(
                df,
                x="Failures_Label",
                y="G3",
                color="Failures_Label",
                title="Impact of Past Course Failures on End Semester Exam",
                labels={"G3": "End Semester Exam Score (0 - 30)"}
            )
            apply_plotly_styling(fig_fail, height=380)
            fig_fail.update_layout(showlegend=False)
            st.plotly_chart(fig_fail, use_container_width=True)

    if "absences" in df.columns and has_g3:
        fig_abs = px.scatter(
            df,
            x="absences",
            y="G3",
            color="Performance_Category",
            color_discrete_map={
                "Low (At-Risk)": "#ef4444",
                "Medium (Satisfactory)": "#f59e0b",
                "High (Good)": "#10b981",
            },
            title="Course Absences vs. End Semester Exam Score",
            labels={"absences": "Number of Course Absences", "G3": "End Semester Exam Score (0 - 30)"},
        )
        apply_plotly_styling(fig_abs, height=360)
        st.plotly_chart(fig_abs, use_container_width=True)

with tab3:
    st.markdown("#### Demographic Indicators & Educational Support")
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        if "higher" in df.columns and has_g3:
            higher_map = {"yes": "Aspires to Higher Ed", "no": "No Higher Ed Aspiration"}
            df["Higher_Label"] = df["higher"].map(higher_map)
            fig_higher = px.histogram(
                df,
                x="G3",
                color="Higher_Label",
                barmode="overlay",
                marginal="box",
                title="Aspiration for Higher Education vs. End Semester Exam",
                labels={"G3": "End Semester Exam Score (0 - 30)"}
            )
            apply_plotly_styling(fig_higher, height=380)
            st.plotly_chart(fig_higher, use_container_width=True)

    with d_col2:
        if "schoolsup" in df.columns and "famsup" in df.columns and has_g3:
            support_df = df.groupby(["schoolsup", "famsup"])["G3"].agg(["mean", "count"]).reset_index()
            support_df["Support_Group"] = "School Support: " + support_df["schoolsup"] + " | Family: " + support_df["famsup"]

            fig_supp = px.bar(
                support_df,
                x="Support_Group",
                y="mean",
                text="mean",
                color="Support_Group",
                title="Mean End Semester Exam Score by Support Structure",
                labels={"mean": "Average End Semester Exam Score (0 - 30)"},
            )
            fig_supp.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            apply_plotly_styling(fig_supp, height=380)
            fig_supp.update_layout(showlegend=False)
            st.plotly_chart(fig_supp, use_container_width=True)

with tab4:
    st.markdown("#### Search and Inspect Dataset Records")
    st.dataframe(df, use_container_width=True, height=400)
    st.caption(f"Showing all {len(df)} records with {len(df.columns)} attributes.")

render_disclaimer()
