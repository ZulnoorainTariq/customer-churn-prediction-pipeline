# Customer Churn Prediction Pipeline

An end-to-end machine learning pipeline that predicts whether a telecom customer is likely to churn (leave the company), built with scikit-learn's `Pipeline` API and deployed with Streamlit.

## What it does

Given details about a customer (contract type, monthly charges, tenure, internet service, etc.), the app predicts whether that customer is likely to churn, along with a probability score.

## Dataset

[Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customer records with 21 features including demographics, account information, and services signed up for.

## Approach

- **Preprocessing**: Built using `ColumnTransformer` inside a scikit-learn `Pipeline`
  - Numeric features scaled with `StandardScaler`
  - Categorical features encoded with `OneHotEncoder`
- **Models tried**: Logistic Regression and Random Forest
- **Hyperparameter tuning**: `GridSearchCV` with 5-fold cross-validation, optimized for F1 score (better suited than accuracy for imbalanced classes)
- **Final model**: Logistic Regression (`C=0.1`, `class_weight='balanced'`, `solver='liblinear'`) — chosen for its higher recall on the churn class, since catching potential churners matters more than avoiding false alarms in this use case
- **Export**: Full pipeline (preprocessing + model) saved as a single `.joblib` file for easy reuse

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 74% |
| Churn Class Recall | 79% |
| Churn Class Precision | 51% |
| Churn Class F1 | 0.62 |

## Tech stack

- Python, pandas, scikit-learn
- Streamlit (deployment)
- joblib (model export)

## How to run locally

\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Files

- `app.py` — Streamlit web app
- `churn_prediction_pipeline.joblib` — trained, exported pipeline
- `requirements.txt` — dependencies
- Notebook — full training and tuning process
