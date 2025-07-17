# Customer Churn Prediction

## Overview
A machine learning project to predict customer churn for a telecom company using the Telco Customer Churn dataset. Built a Random Forest model and deployed it as a Streamlit web app.

## Steps
1. **Data Loading**: Loaded dataset (7,043 rows, 21 columns).
2. **EDA**: Identified high churn for month-to-month contracts (~26.5% for tenure < 40), Electronic check (~45.3%), and fiber optic users. Handled 11 missing TotalCharges.
3. **Preprocessing**: Imputed missing values, encoded categorical features, scaled numerical features.
4. **Model Training**: Trained Random Forest (Accuracy: 0.7878, ROC-AUC: 0.6833). Key features: TotalCharges, tenure, MonthlyCharges.
5. **Evaluation**: Visualized confusion matrices and ROC curves.
6. **Deployment**: Built and deployed Streamlit app: [insert URL].

## Files
- `churn_prediction.ipynb`: Jupyter Notebook with all steps.
- `app.py`: Streamlit app for predictions.
- `churn_model.pkl`: Trained Random Forest model.
- `preprocessor.pkl`: Preprocessing pipeline.
- `requirements.txt`: Dependencies.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run locally: `streamlit run app.py`
3. Access deployed app: [insert URL]

## Results
- Model: Random Forest (Accuracy: 0.7878, Precision: 0.6370, Recall: 0.4611, ROC-AUC: 0.6833).
- Key predictors: tenure, MonthlyCharges, Contract, PaymentMethod.