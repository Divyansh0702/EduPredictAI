# EduPredict AI — Student Performance Prediction System

> **A Machine Learning & Applied AI Web Application for Early Academic Risk Identification and Intervention Planning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.20+-3F4F75.svg)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg)]()

---

## 1. Project Overview

**EduPredict AI** is a machine learning web application engineered to predict student academic performance tiers and proactively flag students who would benefit from early educational interventions. Developed specifically as an applied AI demonstration for academic presentations and internship evaluations, it provides educators, academic counselors, and students with transparent, data-backed insights.

The platform analyzes demographic attributes, weekly study habits, attendance records, school/family educational support, and prior continuous assessments (Mid-Term 1 / ST-1, Mid-Term 2 / ST-2) to classify student **End Semester Exam** outcomes into actionable tiers:
- **Low (At-Risk)**: Below passing standard (End Semester Exam $< 10/20$). Requires immediate academic recovery.
- **Medium (Satisfactory)**: Baseline passing standard ($10 \le \text{End Semester Exam} \le 13/20$).
- **High (Good)**: High standing / distinction pace ($\text{End Semester Exam} \ge 14/20$).

---

## 2. Problem Statement & Objectives

### The Problem
Educational institutions often identify struggling students only after final exams or course failures have already occurred. Traditional retrospective reporting makes proactive remediation difficult, leading to compounded learning deficits, preventable course repeats, and increased dropout rates.

### Core Objectives
1. **Early Academic Intervention**: Predict End Semester Exam performance tiers before final assessments, giving educators an actionable window to intervene.
2. **Algorithm Comparison**: Empirically train and compare three distinct machine learning classifiers (**Logistic Regression**, **Decision Tree**, and **Random Forest**) on held-out test data.
3. **Leakage-Free Modeling**: Strictly eliminate target leakage by isolating the End Semester Exam target ($G3$) from predictive features.
4. **Transparent, Actionable Recommendations**: Generate rule-based educational suggestions tailored to each student's specific risk factors without relying on unexplainable "black-box" conclusions.
5. **Demonstration-Ready Web Application**: Provide an intuitive, responsive Streamlit dashboard supporting single student profiling, batch CSV roster scoring, and instant audit downloads.

---

## 3. Technology Stack

EduPredict AI is built entirely in Python without complex multi-tier dependencies (no React, Spring Boot, or Node.js):

| Component | Technology | Version | Purpose |
|---|---|---|---|
| **Programming Language** | Python | 3.11 / 3.13 | Core runtime environment |
| **Web UI & Dashboard** | Streamlit | $\ge 1.35.0$ | Multipage interactive user interface |
| **Data Processing** | Pandas & NumPy | $\ge 2.2.0$ / $\ge 1.26.0$ | Data loading, manipulation, and transformations |
| **Machine Learning** | scikit-learn | $\ge 1.4.0$ | Pipelines, ColumnTransformer, ML classifiers, evaluation |
| **Interactive Visualizations** | Plotly | $\ge 5.20.0$ | Dynamic charts, heatmaps, box plots, donut charts |
| **Model Serialization** | Joblib | $\ge 1.3.0$ | Pipeline persistence and metadata storage |
| **Testing** | Pytest | $\ge 8.0.0$ | Automated test suite for preprocessing and inference |

---

## 4. Key Features

### 📊 A. Home Overview & KPI Dashboard
- Displays key cohort metrics: Total student records analyzed, average End Semester Exam score, passing rate, and top-performing model accuracy.
- Donut chart depicting the distribution of academic performance tiers.
- Quick-access cards linking directly to all functional pages.

### 📈 B. Exploratory Performance Analytics (`pages/1_Dashboard.py`)
- Course subject toggle: **1. Artificial Intelligence** (`artificial_intelligence_students.csv`) vs. **2. Cloud Computing** (`cloud_computing_students.csv`) or user-uploaded CSV.
- Examination progression box plots (Mid-Term 1 $\rightarrow$ Mid-Term 2 $\rightarrow$ End Semester Exam).
- Interactive study time vs. End Semester Exam performance box plots.
- Impact of past course failures on End Semester Exam scores.
- Course attendance vs. End Semester Exam score scatter plot.
- Educational support matrix (school support $\times$ family support).
- Filterable and searchable raw dataset table.

### 🔮 C. Individual Student Prediction (`pages/2_Predict_Performance.py`)
- Subject selection: **1. Artificial Intelligence** or **2. Cloud Computing**.
- Demonstration presets for instant evaluation during college presentations:
  - *Profile A: At-Risk Student* (Low study time, failing prior mid-terms, 2 failures).
  - *Profile B: Average Student* (Passing mid-terms, moderate study time).
  - *Profile C: High-Achieving Student* (High study time, honors mid-term grades, zero failures).
- Streamlined input form covering core academic track record (ST-1/Mid-Term 1, ST-2/Mid-Term 2, failures, absences, age, study time, support).
- Dedicated minimal optional expander for background factors (Mother & Father Occupation, Home Environment, Family Size).
- End Semester Exam predicted performance badge with risk level coloring.
- Model confidence score and interactive class probability distribution chart.
- Prioritized academic support recommendations (Urgent Interventions, Habit Optimization, Positive Reinforcements).
- Clear advisory disclaimer.

### 🧠 D. Model Benchmarks & Comparison (`pages/3_Model_Analytics.py`)
- Evaluates three algorithms on a held-out test set (20% split, stratified, seed 42):
  1. **Logistic Regression** (L2 Regularized, $C=1.0$)
  2. **Decision Tree Classifier** (Max Depth 5, Min Samples Split 6)
  3. **Random Forest Classifier** (120 Estimators, Max Depth 6)
- Comparison across Accuracy, Weighted F1, Macro F1, Weighted Precision, and Weighted Recall.
- Interactive Confusion Matrix heatmaps with selectable model view.
- Actual vs. predicted category distribution comparison.
- Top 15 most influential feature importance bar chart.
- Documented model limitations and ethical considerations.

### 📂 E. Bulk CSV Prediction & Export (`pages/4_Bulk_Prediction.py`)
- Batch CSV upload supporting comma (`,`) and semicolon (`;`) delimiters.
- One-click **Download Sample CSV Template** pre-populated with example records.
- Automated schema validation flagging missing columns or out-of-range values.
- Cohort risk summary cards and risk distribution pie chart.
- Searchable and filterable prediction results table with confidence scores.
- Single-click **Download Complete Predictions as CSV** for administrative reporting.

---

## 5. Project Directory Structure

```text
EduPredictAI/
├── app.py                         # Main Streamlit application and executive dashboard
├── pages/
│   ├── 1_Dashboard.py             # Cohort exploratory analysis and demographic charts
│   ├── 2_Predict_Performance.py   # Single student prediction form and recommendations
│   ├── 3_Model_Analytics.py       # Model benchmark metrics, confusion matrices, feature importance
│   └── 4_Bulk_Prediction.py       # Batch CSV scoring, validation, and CSV export
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_loader.py             # Data loader, delimiter detection, target categorization
│   ├── preprocessing.py          # scikit-learn ColumnTransformer, feature lists, validation
│   ├── train_model.py             # Model training, test evaluation, model selection, serialization
│   ├── predict.py                 # Single and bulk inference routines, probability scoring
│   ├── evaluate.py                # Comprehensive classification evaluation metrics
│   ├── recommendations.py         # Rule-based transparent academic intervention suggestions
│   └── ui_components.py           # Shared modern CSS styles, headers, metric cards, disclaimers
├── data/
│   ├── student-mat.csv            # UCI Math course dataset (395 records)
│   ├── student-por.csv            # UCI Portuguese course dataset (649 records)
│   ├── student.txt                # Dataset attribute descriptions
│   └── README.md                  # Dataset attribution, schema, and offline placement guide
├── models/
│   ├── .gitkeep                   # Tracks models directory in Git
│   ├── best_model_pipeline.joblib # Serialized fitted winning pipeline
│   ├── all_pipelines.joblib       # Serialized dictionary of all 3 fitted candidate pipelines
│   ├── evaluation_results.joblib  # Serialized evaluation results and metadata
│   └── evaluation_results.json    # Human-readable evaluation metrics and metadata
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py      # Unit tests for data loading, thresholding, leakage checks
│   └── test_prediction.py         # Unit tests for inference, probabilities, and recommendation rules
├── requirements.txt               # Pinned project dependencies
├── README.md                      # Comprehensive project documentation
└── .gitignore                     # Git ignore rules for virtual environments and cache
```

---

## 6. Dataset Attribution & Setup

The primary dataset is the **UCI Student Performance Dataset**:
- **Reference**: https://archive.ics.uci.edu/dataset/320/student+performance
- **Authors**: Paulo Cortez and Alice Silva (University of Minho, Portugal, 2008).

### Dataset Setup Instructions
Both `student-mat.csv` and `student-por.csv` are already packaged in the `data/` directory.

If setting up in a fresh or offline environment:
1. Download the archive from the [UCI Repository](https://archive.ics.uci.edu/dataset/320/student+performance).
2. Extract `student-mat.csv` into `EduPredictAI/data/student-mat.csv`.
3. The application will detect and load it automatically.

---

## 7. Installation & Local Setup (Windows)

Follow these exact commands in **PowerShell** or **Command Prompt**:

### Step 1: Clone or Navigate to the Project Directory
```powershell
cd "d:\Web Develpoment\EduPredictAI"
```

### Step 2: Create and Activate a Python Virtual Environment
```powershell
# Create virtual environment named 'venv'
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# (Or on Windows Command Prompt: venv\Scripts\activate.bat)
```

### Step 3: Upgrade Pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Model Training Instructions

To train and benchmark all three machine learning classifiers, compute test set evaluation metrics, and save serialized pipelines:

```powershell
python src/train_model.py
```

### Expected Output:
```text
Loading dataset from: data/student-mat.csv
Target distribution across classes:
performance_category
Medium (Satisfactory)    165
Low (At-Risk)            130
High (Good)              100
Dataset split: 316 training records, 79 testing records.
Training Logistic Regression...
-> Logistic Regression Results: Accuracy=0.8734, Weighted F1=0.8734, Macro F1=0.8777
Training Decision Tree...
-> Decision Tree Results: Accuracy=0.8481, Weighted F1=0.8498, Macro F1=0.8570
Training Random Forest...
-> Random Forest Results: Accuracy=0.8481, Weighted F1=0.8498, Macro F1=0.8570

Winning Model Selected: 'Logistic Regression' (Highest Weighted F1: 0.8734)
Model artifacts successfully serialized to: models/
```

> **Note:** The Streamlit application also features a one-click **Train Models Now** button in the sidebar and dashboard if model files are ever deleted or need retraining on a new dataset.

---

## 9. Running Automated Tests

Run the full Pytest verification suite:

```powershell
pytest tests/ -v
```

All 11 unit tests verify:
- Delimiter detection (`,` and `;`).
- Grade threshold mapping (`0-14` $\rightarrow$ Low, `15-20` $\rightarrow$ Medium, `21-30` $\rightarrow$ High).
- Target leakage elimination ($G3$ dropped from $X$).
- Pipeline transformation consistency.
- Input validation and missing column rejection.
- Single prediction inference and probability normalization ($\sum p \approx 1.0$).
- Bulk scoring and CSV column generation.
- Recommendation rule triggers and advisory disclaimer presence.

---

## 10. Starting the Streamlit Application

Launch the web application locally:

```powershell
streamlit run app.py
```

Once launched, your browser will open automatically at:
```text
Local URL: http://localhost:8501
Network URL: http://<your-ip>:8501
```

---

## 11. Model Evaluation Methodology

### Target Definition & Leakage Prevention
Student grades are evaluated on a **0 to 30 scale** (e.g., Sessional Tests and End Semester Exam), with **15** serving as the official passing boundary (50%). We convert the final grade (**End Semester Exam** / $G3$) into 3 performance categories:
- **Low (At-Risk)**: $\text{End Semester Exam} < 15$
- **Medium (Satisfactory)**: $15 \le \text{End Semester Exam} \le 20$
- **High (Good)**: $21 \le \text{End Semester Exam} \le 30$

To guarantee zero target leakage, the End Semester Exam ($G3$) is removed completely from input features.

### Train/Test Split
- **Test Size**: 20% held-out test data (79 unseen student records).
- **Stratification**: Class distributions are preserved identically across splits.
- **Reproducibility**: `random_state=42`.

### Empirical Results Comparison (Test Set)
| Model | Test Accuracy | Weighted F1 | Macro F1 | Weighted Precision | Weighted Recall |
|---|---|---|---|---|---|
| **Logistic Regression (Winner)** | **87.34%** | **0.8734** | **0.8777** | **87.34%** | **87.34%** |
| **Random Forest Classifier** | 84.81% | 0.8498 | 0.8570 | 85.34% | 84.81% |
| **Decision Tree Classifier** | 84.81% | 0.8498 | 0.8570 | 85.34% | 84.81% |

**Winning Criterion:** Logistic Regression was selected as the final production model due to achieving the highest weighted F1-score ($0.8734$) and accuracy ($87.34\%$) on the held-out test set, along with robust calibration and interpretable coefficients.

---

## 12. Model Limitations & Ethical Considerations

1. **Contextual & Demographic Scope**: The models are trained on secondary school students from Portugal (2008). Educational practices, grading standards, and sociocultural dynamics vary significantly across global educational environments.
2. **Corpus Size**: The primary dataset comprises 395 mathematics records. Highly unique educational trajectories may have limited representation.
3. **Non-Causal Heuristics**: Academic performance predictions are statistical approximations based on historical correlation. They must **never** be used to deny academic opportunities, gatekeep advanced courses, or assign permanent labels to students.
4. **Human-in-the-Loop Requirement**: All predictions and recommendations are designed strictly as decision-support aids for qualified teachers, advisors, and counselors who understand each student's holistic context.

---

## 13. Deployment

EduPredict AI can be deployed on any standard platform:
- **Streamlit Community Cloud**:
  1. Push this repository to GitHub.
  2. Connect repository to [share.streamlit.io](https://share.streamlit.io).
  3. Specify `app.py` as the main entry file.
- **Docker**:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8501
  CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
  ```

---

## 14. License & Attribution

- **Dataset**: UCI Machine Learning Repository (Cortez & Silva, 2008).
- **Project**: EduPredict AI — Created for Machine Learning and Applied AI Internship Evaluation.

