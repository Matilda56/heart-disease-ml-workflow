# ❤️ Heart Disease Prediction ML Workflow

A production-style end-to-end machine learning pipeline for predicting heart disease using structured clinical data.
This project demonstrates a full ML workflow including data validation, EDA, feature engineering, model training, cross-validation, evaluation, and artifact generation.

Designed as a **portfolio-ready project** for Data Science / Machine Learning roles.

---

## 🚀 Project Overview

This project builds a machine learning system to predict whether a patient has heart disease based on clinical features such as age, cholesterol, ECG results, and more.

The workflow follows a **real-world ML pipeline structure**:

* Data validation
* Exploratory data analysis (EDA)
* Feature preprocessing pipeline
* Model comparison
* Cross-validation
* Final model training
* Evaluation & reporting
* Test prediction generation
* Saved model artifacts

The final system achieves strong predictive performance and is fully reproducible.

---

## 📊 Dataset

**Source:** Kaggle Playground Series
https://www.kaggle.com/competitions/playground-series-s6e2

* ~630,000 training samples
* 15 features (numerical + categorical)
* Binary target: **Heart Disease (0/1)**
* Well-balanced dataset (≈55% vs 45%)

No heavy class imbalance handling required.

---

## 🧠 Key EDA Insights

**Strong predictors**

* ST depression (strong positive correlation)
* Max heart rate (strong negative correlation)
* Number of vessels (Fluro)
* Exercise-induced angina
* Thallium test results
* Chest pain type

**Moderate predictors**

* Age
* Cholesterol

**Weak predictor**

* Blood pressure (BP alone has limited predictive power)

Tree-based and linear models both perform well due to nonlinear relationships.

---

## 🏗️ Project Structure

```
heart-disease-ml-workflow
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── src/
│   ├── config.py                # Paths & global config
│   ├── data_validation.py       # Data quality checks
│   ├── preprocessing.py         # Feature pipelines
│   ├── train.py                 # Model training
│   ├── evaluate.py              # Metrics & evaluation
│   ├── cv.py                    # Cross-validation
│   ├── predict_test.py          # Test prediction generation
│   ├── main.py                  # Full pipeline entry point
│   │
│   ├── models/
│   │   └── logreg_pipeline.pkl  # Saved trained model
│   │
│   └── reports/
│       ├── run_summary.txt      # Training summary
│       └── test_predictions.csv # Predictions
│
├── eda.ipynb                    # Exploratory data analysis
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* Python
* pandas / numpy
* scikit-learn
* joblib
* matplotlib / seaborn
* Jupyter Notebook

Focus: **production-style ML structure**, not just notebook modeling.

---

## 🧪 Models Evaluated

| Model               | ROC-AUC    | Precision | Recall | F1     |
| ------------------- | ---------- | --------- | ------ | ------ |
| Logistic Regression | **0.9537** | 0.8553    | 0.8935 | 0.8740 |
| Random Forest       | 0.9477     | 0.8411    | 0.8945 | 0.8670 |

**Selected best model:** Logistic Regression
(based on validation ROC-AUC)

---

## 📈 Cross Validation (5-Fold)

```
ROC-AUC:   0.9529 ± 0.0004
Precision: 0.8818 ± 0.0005
Recall:    0.8599 ± 0.0020
F1-score:  0.8707 ± 0.0012
```

Model shows strong stability and generalization.

---

## 🔍 Final Model Performance

* ROC-AUC: 0.9537
* F1-score: 0.8740
* Balanced precision/recall
* No severe overfitting
* Stable across folds

Model trained on full dataset and saved as reusable pipeline.

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run full pipeline

```bash
python src/main.py
```

This will:

* Validate data
* Train models
* Run cross-validation
* Evaluate performance
* Save best model
* Generate predictions
* Create run summary report

---

## 📁 Outputs Generated

**Saved model**

```
src/models/logreg_pipeline.pkl
```

**Training summary**

```
src/reports/run_summary.txt
```

**Test predictions**

```
src/reports/test_predictions.csv
```

---

## 🧩 ML Engineering Highlights

This project demonstrates:

✔ End-to-end ML pipeline design
✔ Modular production-style structure
✔ Config-driven paths
✔ Feature preprocessing pipelines
✔ Cross-validation workflow
✔ Model comparison logic
✔ Reproducible training
✔ Artifact saving (model + reports)
✔ Clean GitHub project organization

---

## 🚀 Future Improvements

* Add XGBoost / LightGBM models
* Hyperparameter tuning (Optuna/GridSearch)
* Model explainability (SHAP)
* FastAPI deployment
* Streamlit demo app
* Docker containerization

---

## 👩‍💻 Author

Built as a machine learning portfolio project for data science and ML engineering roles.
