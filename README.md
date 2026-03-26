# Telco Customer Churn Prediction
### Random Forest Classifier with SMOTE & RandomizedSearchCV

A complete end-to-end machine learning pipeline for predicting customer churn, featuring class imbalance handling, hyperparameter optimization, and customer risk segmentation.

---

## Overview

This project builds a churn prediction system on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle). It explores how class imbalance handling and hyperparameter tuning affect model performance, and translates model outputs into actionable customer risk tiers.

**Core objectives:**
- Benchmark Random Forest performance with and without SMOTE
- Optimize hyperparameters via RandomizedSearchCV (F1-score)
- Evaluate end-to-end model improvement across three configurations
- Identify key churn drivers through feature importance analysis
- Segment customers into risk tiers for targeted retention strategies

---

## Dataset

| Property | Detail |
|----------|--------|
| **Source** | Kaggle |
| **Dataset** | [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Target Variable** | `Churn` (binary) |
| **Features** | Customer demographics, account info, and service usage |

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Modelling | `scikit-learn` |
| Imbalance handling | `imbalanced-learn` (SMOTE) |
| Language | Python 3 |

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Null value detection and data type inspection
- Statistical summaries (`describe()`, `info()`)
- Correlation matrix visualisation
- Target distribution analysis to surface class imbalance

### 2. Data Preprocessing
- Missing value imputation
- Categorical encoding via **Target Encoder**
- Feature scaling
- Stratified train-test split

### 3. Handling Class Imbalance
The churn class is underrepresented in the dataset. To address this, **SMOTE** (Synthetic Minority Over-sampling Technique) was applied to the training set.

Models compared:
- Baseline RFC (no resampling)
- RFC + SMOTE

### 4. Hyperparameter Tuning
`RandomizedSearchCV` with 5-fold cross-validation and F1-score as the scoring metric was used to search the following parameter space:

| Parameter | Description |
|-----------|-------------|
| `n_estimators` | Number of trees in the forest |
| `max_depth` | Maximum tree depth |
| `min_samples_split` | Minimum samples required to split a node |
| `min_samples_leaf` | Minimum samples required at a leaf node |
| `class_weight` | Class weighting strategy |
| `bootstrap` | Whether to use bootstrap samples |

### 5. Model Evaluation
Three configurations are benchmarked:

| Model | Description |
|-------|-------------|
| **Baseline RFC** | Default Random Forest, no resampling |
| **RFC + SMOTE** | Random Forest trained on SMOTE-resampled data |
| **Tuned RFC + SMOTE** | SMOTE + RandomizedSearchCV-optimized hyperparameters |

Metrics reported: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

### 6. Feature Importance
The final model produces a feature importance plot identifying the top drivers of churn. Key findings from the SHAP summary:

> `Contract type`, `InternetService`, `tenure`, `OnlineSecurity`, and `TechSupport` are the most influential features. Short tenure, higher monthly charges, and the absence of bundled services are associated with elevated churn risk.

### 7. Customer Risk Segmentation
Predicted churn probabilities are used to categorise customers into three retention tiers:

| Risk Level | Action |
|------------|--------|
| High Risk | Immediate outreach, priority retention offers |
| Medium Risk | Proactive engagement, service upsell |
| Low Risk | Standard communication, loyalty rewards |

---

## Key Insights

- **Class imbalance** materially reduces recall for the churn class when left unaddressed — the model under-detects churners.
- **SMOTE** improves minority class detection at the cost of a slight precision trade-off.
- **Hyperparameter tuning** on top of SMOTE yields the best overall F1-score across both classes.
- Customers with **month-to-month contracts**, **no tech support**, and **short tenure** represent the highest churn risk profile.

---

## Getting Started

This project was developed on Google Colab. No local setup is required.

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and upload it to your Google Drive or Colab session storage.
2. Open the notebook in Google Colab using the button below (or upload it manually via *File → Upload notebook*).
3. Update the dataset file path in the notebook to match your own Drive/storage location.
4. Run all cells in order.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo-name/blob/main/telco_churn_prediction.ipynb)

> **Note:** All required libraries (`scikit-learn`, `imbalanced-learn`, etc.) are pre-installed in the Colab runtime and do not require manual installation.

---

## Acknowledgements

Dataset provided by [IBM via Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
