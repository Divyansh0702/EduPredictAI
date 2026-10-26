# Dataset Documentation: Student Performance Dataset

## 1. Overview and Attribution
The dataset used in **EduPredict AI** is the well-known **Student Performance Dataset** hosted on the UCI Machine Learning Repository.

- **Source Reference**: https://archive.ics.uci.edu/dataset/320/student+performance
- **Citation**: 
  > P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance*. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.

## 2. Dataset Files According to Subjects
- `artificial_intelligence_students.csv`: Student performance records for the **Artificial Intelligence** course (395 records).
- `cloud_computing_students.csv`: Student performance records for the **Cloud Computing** course (649 records).
- `student-mat.csv`: Original UCI source file for Mathematics.
- `student-por.csv`: Original UCI source file for Portuguese.
- `student.txt`: Detailed attribute metadata description provided by the authors.

## 3. Delimiter and Formatting
- The subject CSV files (`artificial_intelligence_students.csv` and `cloud_computing_students.csv`) use standard **comma (`,`)** delimiters and include an explicit `subject` column.
- The original UCI files use a **semicolon (`;`)** delimiter.
- EduPredict AI's `src.data_loader` includes automatic delimiter detection (`detect_delimiter`) supporting both comma (`,`) and semicolon (`;`) delimited files seamlessly.

## 4. Feature Schema
The dataset contains student attributes:
- **Demographic & Family**: `school`, `sex`, `age`, `address`, `famsize`, `Pstatus`, `Medu`, `Fedu`, `Mjob`, `Fjob`, `reason`, `guardian`
- **Academic & School Support**: `traveltime`, `studytime`, `failures`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`
- **Social & Health**: `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`
- **Academic Assessments**:
  - `G1`: Mid-Term 1 / Sessional Test 1 score (numeric: 0 to 30).
  - `G2`: Mid-Term 2 / Sessional Test 2 score (numeric: 0 to 30).
  - `G3`: **End Semester Exam** score (numeric: 0 to 30, target variable).

## 5. Target Variable Definition & Leakage Prevention
To prevent **target leakage**:
1. The target variable `performance_category` is computed directly from the **End Semester Exam** (`G3`).
2. The raw `G3` column is strictly removed from feature matrix `X` before training and testing.
3. Only historical features and continuous internal assessments (`G1`, `G2`) are made available to the model.

### Educational Grading Thresholds (0 - 30 Scale)
| End Semester Exam (G3) | Performance Category | Educational Interpretation | Recommended Support Level |
|---|---|---|---|
| **0 to 14** | **Low (At-Risk)** | Below minimum passing standard (< 15 / 30) | Immediate targeted remediation & tutoring |
| **15 to 20** | **Medium (Satisfactory)** | Standard passing grade; meets baseline goals (15 - 20 / 30) | Habit optimization & active recall |
| **21 to 30** | **High (Good)** | Above-average to distinction pace (>= 21 / 30) | Advanced projects & academic leadership |

## 6. Manual Dataset Setup Instructions
If offline or downloading manually:
1. Download the ZIP file from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance).
2. Extract `student-mat.csv` and `student-por.csv`.
3. Place them directly inside the `data/` folder of this project:
   ```text
   EduPredictAI/data/student-mat.csv
   EduPredictAI/data/student-por.csv
   ```
4. Run the training script:
   ```powershell
   python src/train_model.py
   ```

