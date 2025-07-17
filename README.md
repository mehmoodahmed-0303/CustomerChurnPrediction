Data Science Projet =========   Customer Churn Prediction using Machine learning




# Customer Churn Prediction Project

## Overview
This project predicts customer churn for a telecom company using the Telco Customer Churn dataset. The goal is to identify at-risk customers and build a deployable machine learning model. The workflow includes data loading, exploratory data analysis (EDA), preprocessing, model training, and evaluation.

## Step 1: Data Loading and Inspection
- **Objective**: Load and inspect the Telco Customer Churn dataset.
- **Actions**:
  - Loaded dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) with 7,043 rows and 21 columns (e.g., `customerID`, `tenure`, `MonthlyCharges`, `Churn`).
  - Inspected structure: Mix of numerical (`tenure`, `MonthlyCharges`, `TotalCharges`) and categorical (`Contract`, `PaymentMethod`) features.
  - Identified 11 missing values in `TotalCharges` (empty strings).
- **Findings**:
  - `TotalCharges` stored as object, needs conversion to numeric.
  - No other missing values.
- **Tools**: Python, pandas.

## Step 2: Exploratory Data Analysis (EDA)
- **Objective**: Analyze patterns in numerical and categorical features related to churn.
- **Actions**:
  - Converted `TotalCharges` to numeric, confirming 11 missing values.
  - Created **histograms** for `tenure` (right-skewed, peak at 0–10 months), `MonthlyCharges` (bimodal, peaks at ~$20 and ~$70–100), and `TotalCharges` (right-skewed).
  - Created **box plots** to compare numerical features by `Churn` (Yes/No).
  - Created **bar plots** for categorical features (`gender`, `Contract`, `PaymentMethod`, `InternetService`) by `Churn`.
  - Computed **correlation heatmap** for numerical features and `Churn_Encoded` (Yes=1, No=0).
- **Findings**:
  - **Numerical**: Churners have lower `tenure` (median ~10 months vs. ~40 for non-churners), higher `MonthlyCharges` (~$80 vs. ~$60), lower `TotalCharges`.
  - **Categorical**: Month-to-month contracts (~40% churn), fiber optic `InternetService`, and `Electronic check` (~45.3% churn) have high churn rates. `gender` neutral.
  - **Specific**: 1,869 churners, 5,174 non-churners with `tenure < 40` and month-to-month contracts (~26.5% churn).
  - **PaymentMethod**: 
    - Electronic check: 1,071 churners, 1,294 non-churners (~45.3% churn).
    - Mailed check: 308 churners, 1,304 non-churners (~19.1% churn).
    - Bank transfer: 258 churners, 1,286 non-churners (~16.7% churn).
    - Credit card: 232 churners, 1,290 non-churners (~15.3% churn).
  - **Correlations**: `tenure` (~-0.35, negative), `MonthlyCharges` (~0.20, positive) with churn.
- **Tools**: pandas, Matplotlib, Seaborn.

## Step 3: Data Preprocessing
- **Objective**: Prepare data for modeling by handling missing values, encoding categorical variables, and scaling numerical features.
- **Actions**:
  - Imputed 11 missing `TotalCharges` with median.
  - Encoded categorical variables (e.g., `Contract`, `PaymentMethod`) using OneHotEncoder.
  - Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using StandardScaler.
  - Dropped `customerID` (non-predictive).
  - Created preprocessing pipeline with `ColumnTransformer`.
  - Split data: 80% train (~5,634 rows), 20% test (~1,409 rows).
  - Saved preprocessor (`preprocessor.pkl`) for deployment.
- **Findings**:
  - Preprocessed data shape: ~43 features after encoding (numerical + one-hot-encoded categorical).
  - Pipeline ensures consistent preprocessing for training and deployment.
- **Tools**: pandas, scikit-learn.

## Step 4: Model Building and Training
- **Objective**: Train and evaluate machine learning models to predict churn.
- **Actions**:
  - Trained Logistic Regression and Random Forest models.
  - Evaluated using accuracy, precision, recall, and ROC-AUC.
  - Computed Random Forest feature importance.
  - Saved Random Forest model (`churn_model.pkl`) and preprocessor for deployment.
- **Findings**:
  - Performance: [To be filled with your output, e.g., Random Forest: Accuracy 0.79, ROC-AUC 0.74].
  - Top features: [To be filled, e.g., `tenure`, `Contract_Month-to-month`, `MonthlyCharges`, `PaymentMethod_Electronic check`].
  - Key predictors align with EDA (short tenure, month-to-month contracts, electronic check).
- **Tools**: scikit-learn, joblib.

## Step 5: Model Training
- Trained Logistic Regression and Random Forest.
- Performance: [e.g., Random Forest: Accuracy 0.79, ROC-AUC 0.74].
- Top features: [e.g., tenure, Contract_Month-to-month, MonthlyCharges, PaymentMethod_Electronic check].
- Saved model and preprocessor for deployment.


## Model Evaluation
- Visualized confusion matrices and ROC curves.
- Performance: [e.g., Random Forest: Accuracy 0.79, ROC-AUC 0.74].
- Confusion Matrix: [e.g., Random Forest: High TP, low FN].
- ROC AUC: [e.g., Random Forest: 0.74, Logistic Regression: 0.80].
- Selected Random Forest for deployment.