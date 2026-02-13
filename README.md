Telco Customer Churn Prediction – Random Forest with SMOTE & RandomizedSearchCV
📌 Project Overview

This project investigates customer churn prediction using a Random Forest Classifier (RFC) on the Telco Customer Churn dataset from Kaggle.

The primary objective is to:

Compare Random Forest performance with and without SMOTE

Optimize hyperparameters using RandomizedSearchCV

Evaluate model performance improvements

Analyze feature importance

Segment customers into High / Medium / Low churn risk

This project demonstrates an end-to-end machine learning workflow including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and interpretation.

📂 Dataset

Source: Kaggle

Dataset: Telco Customer Churn

Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains customer demographic, account, and service usage information, with a binary churn target variable.

🛠️ Technologies Used

Python 3.x

Pandas

NumPy

Matplotlib / Seaborn

Scikit-learn

imbalanced-learn (SMOTE)

🔎 Project Workflow
1️⃣ Exploratory Data Analysis (EDA)

Basic EDA steps include:

Null value checking

Data type inspection (info())

Statistical summaries (describe())

Correlation matrix visualization

Target distribution analysis (churn imbalance)

2️⃣ Data Preprocessing

Handling missing values

Encoding categorical variables

Feature scaling 

Train-test split

3️⃣ Handling Class Imbalance

The dataset is imbalanced.

To address this:

Applied SMOTE (Synthetic Minority Over-sampling Technique)

Compared model performance:

RFC without SMOTE

RFC with SMOTE

4️⃣ Model Optimization

Hyperparameter tuning performed using:

RandomizedSearchCV with F1-score scoring and cross validation

Optimized parameters for:

n_estimators

max_depth

min_samples_split

min_samples_leaf

class_weight

bootstrap

5️⃣ Model Evaluation

Evaluation metrics include:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Performance comparison between:

Baseline RFC

RFC + SMOTE

Tuned RFC + SMOTE

6️⃣ Feature Importance

The final model produces a feature importance visualization, identifying key drivers of customer churn.

This helps improve interpretability and supports business decision-making.

7️⃣ Customer Risk Segmentation

Based on predicted churn probabilities, customers are categorized into:

🔴 High Risk

🟠 Medium Risk

🟢 Low Risk

This segmentation enables targeted retention strategies.

📈 Key Insights

Class imbalance significantly affects recall for churn prediction.

SMOTE improves detection of minority (churn) class.

Hyperparameter tuning further enhances model performance.

Certain service-related and contract features are strong churn predictors.
