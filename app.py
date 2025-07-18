import streamlit as st
import pandas as pd
import joblib

print("Streamlit app starting...")  # Add this
# Load model and preprocessor
model = joblib.load('Churn_model.pkl')
preprocessor = joblib.load('Preprocessor.pkl')

# Streamlit app
st.title("Customer Churn Prediction")

# Input fields
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 600.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

# Prepare input data
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# Predict
if st.button("Predict"):
    input_preprocessed = preprocessor.transform(input_data)
    prediction = model.predict(input_preprocessed)[0]
    probability = model.predict_proba(input_preprocessed)[0][1]
    st.write(f"Churn Prediction: {'Yes' if prediction == 1 else 'No'}")
    st.write(f"Churn Probability: {probability:.2%}")
