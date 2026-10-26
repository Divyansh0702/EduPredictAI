"""
Page 3: Model Performance Analytics & Algorithm Comparison
Displays test-set evaluation metrics, algorithm comparisons, confusion matrices,
feature importances, and methodology documentation.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import CLASS_NAMES
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

st.set_page_config(
    page_title="Model Analytics — EduPredict AI",
    page_icon="🧠",
    layout="wide",
)

apply_custom_css()

# Load evaluation results
if not is_model_trained():
    render_sidebar_brand(model_ready=False)
    st.warning("⚠️ No evaluated model artifacts found on disk.")
    st.info("Click the button below to train and benchmark all 3 classifiers on the dataset:")
    if st.button("🚀 Train & Benchmark Models", key="train_analytics"):
        with st.spinner("Training and evaluating models..."):
            train_and_evaluate_all()
            st.success("Training and evaluation completed!")
            st.rerun()
    st.stop()

try:
    eval_payload = load_evaluation_results()
    metadata = eval_payload["metadata"]
    models_dict = eval_payload["models"]
    feature_importances = eval_payload.get("feature_importances", {})
    winning_model = metadata["winning_model"]
    winning_acc = models_dict[winning_model]["accuracy"] * 100
    render_sidebar_brand(model_ready=True, winning_model=winning_model, accuracy_str=f"{winning_acc:.1f}%")
except Exception as e:
    st.error(f"Failed to load evaluation results: {e}")
    st.stop()

render_header(
    title="Machine Learning Model Benchmarks & Analytics",
    subtitle=(
        "Rigorous comparative evaluation of Logistic Regression, Decision Tree, and Random Forest "
        "classifiers on held-out test data. Inspect accuracy, F1-scores, confusion matrices, and feature importances."
    ),
    icon="🧠",
    badge="Module 3: Algorithm Benchmarks & Verification"
)

# Methodology Alert Banner
st.markdown(
    f"""
    <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 14px 20px; margin-bottom: 24px;">
        <span style="font-weight: 700; color: #166534; font-size: 1rem;">📋 Evaluation Protocol & Methodology:</span>
        <div style="font-size: 0.88rem; color: #15803d; margin-top: 4px;">
            • <strong>Split:</strong> 80% Training ({metadata['train_samples']} samples) / 20% Held-Out Testing ({metadata['test_samples']} samples), Stratified.<br>
            • <strong>Target:</strong> 3-Tier Classification of End Semester Exam (Low &lt; 15, Medium 15–20, High &ge; 21 on a 0–30 scale). Strict isolation: End Semester Exam (G3) removed from features.<br>
            • <strong>Reproducibility:</strong> Fixed Random Seed = {metadata['random_seed']}.<br>
            • <strong>Selection Metric:</strong> {metadata['selection_criterion']} &rarr; <strong>Winning Model: {metadata['winning_model']}</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Section 1: Comparative Metric Cards
st.markdown("### 🏆 Algorithm Benchmark Summary (Held-out Test Set)")

comp_data = []
for model_name, metrics in models_dict.items():
    comp_data.append({
        "Algorithm": model_name,
        "Accuracy": metrics["accuracy"],
        "Weighted F1": metrics["f1_weighted"],
        "Macro F1": metrics["f1_macro"],
        "Precision (W)": metrics["precision_weighted"],
        "Recall (W)": metrics["recall_weighted"],
    })
comp_df = pd.DataFrame(comp_data)

# Display styled metrics table
formatted_df = comp_df.copy()
for col in ["Accuracy", "Weighted F1", "Macro F1", "Precision (W)", "Recall (W)"]:
    formatted_df[col] = formatted_df[col].apply(lambda x: f"{x * 100:.2f}%")

st.dataframe(
    formatted_df,
    use_container_width=True,
    hide_index=True,
)

# Section 2: Interactive Model Comparison Chart
st.markdown("### 📊 Multi-Metric Comparison Chart")
chart_df = comp_df.melt(
    id_vars=["Algorithm"],
    value_vars=["Accuracy", "Weighted F1", "Macro F1", "Precision (W)", "Recall (W)"],
    var_name="Metric",
    value_name="Score"
)
chart_df["Score_Pct"] = chart_df["Score"] * 100

fig_comp = px.bar(
    chart_df,
    x="Metric",
    y="Score_Pct",
    color="Algorithm",
    barmode="group",
    text="Score_Pct",
    color_discrete_sequence=["#3b82f6", "#10b981", "#8b5cf6"],
    title="Comparison Across Accuracy, F1-Scores, Precision, and Recall (Test Data)",
    labels={"Score_Pct": "Score (%)"},
)
fig_comp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
apply_plotly_styling(fig_comp, height=400)
fig_comp.update_layout(yaxis=dict(range=[0, 110]))
st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section 3: Confusion Matrix & Actual vs Predicted Analysis
st.markdown("### 🎯 Confusion Matrix & Class Predictions")
cm_col1, cm_col2 = st.columns([1.1, 0.9])

with cm_col1:
    selected_model_name = st.selectbox(
        "Select Model to Inspect Confusion Matrix:",
        options=list(models_dict.keys()),
        index=list(models_dict.keys()).index(metadata["winning_model"]),
    )

    selected_eval = models_dict[selected_model_name]
    cm_matrix = np.array(selected_eval["confusion_matrix"])

    fig_cm = px.imshow(
        cm_matrix,
        labels=dict(x="Predicted Performance", y="Actual Performance", color="Students"),
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"Confusion Matrix: {selected_model_name} (Test Set)",
    )
    apply_plotly_styling(fig_cm, height=360)
    st.plotly_chart(fig_cm, use_container_width=True)

with cm_col2:
    st.markdown(f"#### Actual vs. Predicted Breakdown ({selected_model_name})")
    act_counts = selected_eval["actual_distribution"]
    pred_counts = selected_eval["predicted_distribution"]

    breakdown_df = pd.DataFrame([
        {"Category": cat, "Actual (Ground Truth)": act_counts.get(cat, 0), "Predicted by Model": pred_counts.get(cat, 0)}
        for cat in CLASS_NAMES
    ])

    fig_breakdown = go.Figure(data=[
        go.Bar(name="Actual (Ground Truth)", x=breakdown_df["Category"], y=breakdown_df["Actual (Ground Truth)"], marker_color="#94a3b8"),
        go.Bar(name="Predicted by Model", x=breakdown_df["Category"], y=breakdown_df["Predicted by Model"], marker_color="#3b82f6"),
    ])
    apply_plotly_styling(fig_breakdown, height=360)
    fig_breakdown.update_layout(barmode="group", title="Class Volume: Actual vs. Predicted on Test Set")
    st.plotly_chart(fig_breakdown, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section 4: Feature Importance Analysis
st.markdown("### 🔍 Feature Importance & Influential Predictors")
st.markdown(
    "Examine which academic, demographic, and behavioral features carry the greatest weight "
    "in determining student performance classifications."
)

feat_model_choice = st.selectbox(
    "Inspect Feature Importance For:",
    options=[m for m in feature_importances.keys()],
    index=0,
)

if feat_model_choice in feature_importances:
    raw_imps = feature_importances[feat_model_choice]
    # Take top 15 features
    top_items = list(raw_imps.items())[:15]
    top_df = pd.DataFrame(top_items, columns=["Feature", "Relative Importance"]).sort_values(by="Relative Importance", ascending=True)

    fig_imp = px.bar(
        top_df,
        x="Relative Importance",
        y="Feature",
        orientation="h",
        text="Relative Importance",
        color="Relative Importance",
        color_continuous_scale="Viridis",
        title=f"Top 15 Most Influential Features — {feat_model_choice}",
    )
    fig_imp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    apply_plotly_styling(fig_imp, height=450)
    fig_imp.update_layout(showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.info(
        "💡 **Key Analytical Finding:** Across all models, **prior assessment scores (Mid-Term 2 and Mid-Term 1 / ST-2 and ST-1)**, "
        "**past course failures**, and **course absences** emerge consistently as the top predictive signals for the **End Semester Exam** outcome."
    )

st.markdown("<br>", unsafe_allow_html=True)

# Section 5: Model Limitations & Ethical Considerations
st.markdown("### ⚖️ Documented Model Limitations & Ethical Guidance")
lims = metadata.get("limitations", {})

lim_col1, lim_col2 = st.columns(2)
with lim_col1:
    st.markdown(
        f"""
        <div class="metric-card" style="height: 160px;">
            <div class="metric-title">Demographic & Contextual Bias</div>
            <div style="font-size: 0.88rem; color: #475569; margin-top: 6px; line-height: 1.45;">
                {lims.get('dataset_bias', 'Trained on Portuguese secondary students. Sociocultural dynamics may differ in other regions.')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="metric-card" style="height: 160px; margin-top: 14px;">
            <div class="metric-title">Sample Corpus Constraints</div>
            <div style="font-size: 0.88rem; color: #475569; margin-top: 6px; line-height: 1.45;">
                {lims.get('sample_size', 'Sample size reflects 395 records. Highly unique educational trajectories may have limited representation.')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with lim_col2:
    st.markdown(
        f"""
        <div class="metric-card" style="height: 160px;">
            <div class="metric-title">Feature Dependency</div>
            <div style="font-size: 0.88rem; color: #475569; margin-top: 6px; line-height: 1.45;">
                {lims.get('feature_availability', 'Predictive precision relies heavily on prior period grades (G1, G2).')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="metric-card" style="height: 160px; margin-top: 14px;">
            <div class="metric-title">Non-Deterministic Human Factors</div>
            <div style="font-size: 0.88rem; color: #475569; margin-top: 6px; line-height: 1.45;">
                {lims.get('non_deterministic_human_factors', 'Predictions are statistical approximations, not absolute measures of potential.')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_disclaimer()

