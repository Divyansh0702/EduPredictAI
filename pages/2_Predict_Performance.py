"""
Page 2: Student Performance Prediction & Academic Support Recommendations
Interactive form for individual student evaluation, confidence probabilities,
and tailored educational guidance.
"""

import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import is_model_trained, load_best_pipeline, load_evaluation_results, predict_single
from src.preprocessing import FEATURE_METADATA
from src.recommendations import generate_academic_recommendations
from src.train_model import train_and_evaluate_all
from src.ui_components import (
    apply_custom_css,
    apply_plotly_styling,
    render_disclaimer,
    render_header,
    render_sidebar_brand,
)

st.set_page_config(
    page_title="Predict Performance — EduPredict AI",
    page_icon="🔮",
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
    title="Individual Student Performance Forecast",
    subtitle=(
        "Enter student sessional marks (ST-1, ST-2), study habits, and attendance to forecast "
        "their End Semester Exam outcome and unlock tailored academic intervention plans."
    ),
    icon="🔮",
    badge="Module 2: Real-Time Inference Engine"
)

# Model Availability Check
if not is_model_trained():
    st.error("⚠️ No trained model pipeline found on disk.")
    st.info("The application requires trained model weights to make predictions. Please initialize training:")
    if st.button("🚀 Train Models Now (One-Click)", key="train_pred_page"):
        with st.spinner("Training models on dataset..."):
            train_and_evaluate_all()
            st.success("Model trained successfully!")
            st.rerun()
    st.stop()

# Load fitted pipeline
try:
    pipeline = load_best_pipeline()
except Exception as e:
    st.error(f"Error loading trained pipeline: {e}")
    st.stop()

# Preset Profile Loader for live demonstrations
st.markdown("### ⚡ Quick-Load Demonstration Presets")
preset_choice = st.selectbox(
    "Choose a pre-filled profile or configure custom parameters below:",
    options=[
        "Custom Input (Configure Manually)",
        "Profile A: At-Risk Student (Low study time, failing prior grades, 2 past failures)",
        "Profile B: Moderate/Average Student (Passing grades, 2-5 hrs study)",
        "Profile C: High-Achieving Student (Honors grades, high study time, zero failures)",
    ],
    index=0,
)

# Default values based on preset
defaults = {}
if "Profile A" in preset_choice:
    defaults = {
        "studytime": 1, "failures": 2, "G1": 9, "G2": 10, "ST1": 9, "ST2": 10, "absences": 14,
        "schoolsup": "no", "famsup": "no", "age": 18
    }
elif "Profile B" in preset_choice:
    defaults = {
        "studytime": 2, "failures": 0, "G1": 17, "G2": 17, "ST1": 17, "ST2": 17, "absences": 4,
        "schoolsup": "no", "famsup": "yes", "age": 18
    }
elif "Profile C" in preset_choice:
    defaults = {
        "studytime": 4, "failures": 0, "G1": 26, "G2": 27, "ST1": 26, "ST2": 27, "absences": 2,
        "schoolsup": "no", "famsup": "yes", "age": 17
    }

# Input Form
with st.form("student_prediction_form"):
    st.markdown("### 📝 Student Evaluation Form")

    # Subject selection as requested: 1. Artificial Intelligence 2. Cloud Computing
    subject_val = st.selectbox(
        "Select Subject / Course:",
        options=["1. Artificial Intelligence", "2. Cloud Computing"],
        index=0,
        help="Select the academic course being evaluated."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📚 Academic Record")
        g1_val = st.number_input(
            "Sessional Test - 1 (ST-1: 0 - 30)",
            min_value=0, max_value=30,
            value=defaults.get("ST1", defaults.get("G1", 15)),
            help="Sessional Test 1 score (0-30 scale)"
        )
        g2_val = st.number_input(
            "Sessional Test - 2 (ST-2: 0 - 30)",
            min_value=0, max_value=30,
            value=defaults.get("ST2", defaults.get("G2", 15)),
            help="Sessional Test 2 score (0-30 scale)"
        )
        failures_val = st.selectbox(
            "Past Course Failures",
            options=[0, 1, 2, 3],
            index=[0, 1, 2, 3].index(defaults.get("failures", 0)),
            format_func=lambda x: f"{x} failures" if x < 3 else "3 or more failures",
            help="Number of backlogs or failed attempts in prior courses"
        )
        absences_val = st.number_input(
            "Total Course Absences (days)",
            min_value=0, max_value=93,
            value=defaults.get("absences", 4),
            help="Accumulated missed lectures/classes"
        )

    with col2:
        st.markdown("#### ⏱️ Study Habits & Support")
        age_val = st.slider(
            "Student Age (years)",
            min_value=15, max_value=25,
            value=defaults.get("age", 18),
            help="Student age in years"
        )
        study_options = [1, 2, 3, 4]
        study_map = {1: "< 2 hours/wk", 2: "2 - 5 hours/wk", 3: "5 - 10 hours/wk", 4: "> 10 hours/wk"}
        studytime_val = st.selectbox(
            "Weekly Independent Study Time",
            options=study_options,
            index=study_options.index(defaults.get("studytime", 2)),
            format_func=lambda x: study_map[x],
            help="Weekly hours dedicated to self-study"
        )
        schoolsup_val = st.radio(
            "Extra Academic Support from Department/School?",
            options=["no", "yes"],
            index=["no", "yes"].index(defaults.get("schoolsup", "no")),
            format_func=lambda x: "Yes (Receiving Extra Support)" if x == "yes" else "No Support",
            horizontal=True
        )
        famsup_val = st.radio(
            "Family Educational Support?",
            options=["no", "yes"],
            index=["no", "yes"].index(defaults.get("famsup", "yes")),
            format_func=lambda x: "Yes (Family Support)" if x == "yes" else "No Support",
            horizontal=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Optional section: Only minimum parameters (Mother and Father occupation, home environment, family size)
    with st.expander("⚙️ Additional Background Factors (Optional)", expanded=False):
        st.caption("These background parameters are optional. Sensible defaults are used if left unchanged.")
        opt_col1, opt_col2 = st.columns(2)

        with opt_col1:
            mjob_val = st.selectbox(
                "Mother's Occupation",
                options=["other", "at_home", "health", "services", "teacher"],
                format_func=lambda x: {
                    "other": "Other",
                    "at_home": "At Home",
                    "health": "Health Care",
                    "services": "Civil Services",
                    "teacher": "Teacher",
                }.get(x, x),
                index=0,
            )
            fjob_val = st.selectbox(
                "Father's Occupation",
                options=["other", "at_home", "health", "services", "teacher"],
                format_func=lambda x: {
                    "other": "Other",
                    "at_home": "At Home",
                    "health": "Health Care",
                    "services": "Civil Services",
                    "teacher": "Teacher",
                }.get(x, x),
                index=0,
            )

        with opt_col2:
            address_choice = st.selectbox(
                "Home Environment",
                options=["Urban", "Rural"],
                index=0,
                help="Urban or rural residential area"
            )
            address_val = "U" if address_choice == "Urban" else "R"

            famsize_choice = st.selectbox(
                "Family Size",
                options=["> 3 Members", "<= 3 Members"],
                index=0,
                help="Household family size"
            )
            famsize_val = "GT3" if famsize_choice == "> 3 Members" else "LE3"

    submitted = st.form_submit_button("🔮 Predict End Semester Exam Performance", use_container_width=True, type="primary")

if submitted:
    # Build complete student input payload
    student_payload = {
        # Core inputs (0-30 sessional test scores matching model training scale)
        "age": int(age_val),
        "studytime": int(studytime_val),
        "failures": int(failures_val),
        "absences": int(absences_val),
        "G1": int(g1_val),
        "G2": int(g2_val),
        "schoolsup": schoolsup_val,
        "famsup": famsup_val,

        # Optional inputs
        "Mjob": mjob_val,
        "Fjob": fjob_val,
        "address": address_val,
        "famsize": famsize_val,

        # Hidden standard defaults (attending college removed, other background factors streamlined)
        "school": "GP",
        "sex": "F",
        "Pstatus": "T",
        "Medu": 2,
        "Fedu": 2,
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 1,
        "health": 4,
    }

    try:
        result = predict_single(pipeline, student_payload)
        prediction = result["prediction"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]
        risk_level = result["risk_level"]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## 🎯 End Semester Exam Prediction & Risk Analysis")

        # Result badge and probability columns
        res_col1, res_col2 = st.columns([1, 1.2])

        with res_col1:
            badge_class = "badge-good"
            if "Low" in prediction:
                badge_class = "badge-at-risk"
            elif "Medium" in prediction:
                badge_class = "badge-satisfactory"

            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 5px solid {result['risk_color']};">
                    <div class="metric-title">{subject_val} — End Semester Exam Forecast</div>
                    <div style="font-size: 1.85rem; font-weight: 800; color: #0f172a; margin-top: 10px;">
                        <span class="{badge_class}">{prediction}</span>
                    </div>
                    <div style="margin-top: 16px; font-size: 0.95rem; font-weight: 600; color: #334155; display: flex; align-items: center; justify-content: space-between;">
                        <span>Status: <strong style="color: {result['risk_color']};">{risk_level}</strong></span>
                        <span>Confidence: <strong>{confidence * 100:.1f}%</strong></span>
                    </div>
                    <div style="background-color: #f1f5f9; border-radius: 9999px; height: 8px; width: 100%; margin-top: 8px; overflow: hidden;">
                        <div style="background-color: {result['risk_color']}; width: {confidence * 100:.1f}%; height: 100%; border-radius: 9999px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            if probabilities:
                st.markdown("##### Class Probability Distribution")
                prob_df = [
                    {"Category": cat, "Probability": round(prob * 100, 1)}
                    for cat, prob in probabilities.items()
                ]
                fig_prob = px.bar(
                    prob_df,
                    x="Probability",
                    y="Category",
                    orientation="h",
                    text="Probability",
                    color="Category",
                    color_discrete_map={
                        "Low (At-Risk)": "#ef4444",
                        "Medium (Satisfactory)": "#f59e0b",
                        "High (Good)": "#10b981",
                    },
                )
                fig_prob.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                apply_plotly_styling(fig_prob, height=210)
                fig_prob.update_layout(
                    margin=dict(l=10, r=45, t=10, b=10),
                    xaxis=dict(range=[0, 115], title="Probability (%)"),
                    yaxis=dict(title=""),
                    showlegend=False,
                )
                st.plotly_chart(fig_prob, use_container_width=True)

        # Academic Support Recommendations Section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💡 Tailored Academic Support Recommendations")

        recs = generate_academic_recommendations(student_payload, prediction)

        # Urgent Actions
        if recs["urgent_actions"]:
            st.markdown("#### 🚨 Priority Interventions (High Attention)")
            for item in recs["urgent_actions"]:
                st.markdown(
                    f"""
                    <div class="rec-card-urgent">
                        <div style="font-weight: 700; color: #9f1239; font-size: 1rem;">
                            📌 {item['title']} <span style="font-size: 0.75rem; background: #ffe4e6; padding: 2px 8px; border-radius: 4px; margin-left: 8px;">{item['action_tag']}</span>
                        </div>
                        <div style="font-size: 0.82rem; color: #be123c; margin: 4px 0;">Triggered by: {item['indicator']}</div>
                        <div style="font-size: 0.9rem; color: #4c0519; margin-top: 6px;">{item['suggestion']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Recommended Actions
        if recs["recommended_actions"]:
            st.markdown("#### 📋 Recommended Habit & Support Optimizations")
            for item in recs["recommended_actions"]:
                st.markdown(
                    f"""
                    <div class="rec-card-recommended">
                        <div style="font-weight: 700; color: #1e40af; font-size: 1rem;">
                            🔧 {item['title']} <span style="font-size: 0.75rem; background: #dbeafe; padding: 2px 8px; border-radius: 4px; margin-left: 8px;">{item['action_tag']}</span>
                        </div>
                        <div style="font-size: 0.82rem; color: #1d4ed8; margin: 4px 0;">Triggered by: {item['indicator']}</div>
                        <div style="font-size: 0.9rem; color: #1e3a8a; margin-top: 6px;">{item['suggestion']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Positive Notes
        if recs["positive_notes"]:
            st.markdown("#### 🌟 Commendable Strengths & Accelerators")
            for item in recs["positive_notes"]:
                st.markdown(
                    f"""
                    <div class="rec-card-positive">
                        <div style="font-weight: 700; color: #065f46; font-size: 1rem;">
                            ✨ {item['title']}
                        </div>
                        <div style="font-size: 0.82rem; color: #047857; margin: 4px 0;">Observed Indicator: {item['indicator']}</div>
                        <div style="font-size: 0.9rem; color: #064e3b; margin-top: 6px;">{item['suggestion']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error(f"Prediction execution failed: {e}")

render_disclaimer()

