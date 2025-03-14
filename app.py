import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from pathlib import Path

# Define the expected features (these should match the training data after one-hot encoding)
expected_features = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "job_admin.",
    "job_blue-collar",
    "job_entrepreneur",
    "job_housemaid",
    "job_management",
    "job_retired",
    "job_self-employed",
    "job_services",
    "job_student",
    "job_technician",
    "job_unemployed",
    "job_unknown",
    "marital_divorced",
    "marital_married",
    "marital_single",
    "education_primary",
    "education_secondary",
    "education_tertiary",
    "education_unknown",
    "default_no",
    "default_yes",
    "housing_no",
    "housing_yes",
    "loan_no",
    "loan_yes",
    "contact_cellular",
    "contact_telephone",
    "contact_unknown",
    "month_apr",
    "month_aug",
    "month_dec",
    "month_feb",
    "month_jan",
    "month_jul",
    "month_jun",
    "month_mar",
    "month_may",
    "month_nov",
    "month_oct",
    "month_sep",
    "poutcome_failure",
    "poutcome_other",
    "poutcome_success",
    "poutcome_unknown"
]

# Define the path to your trained model
MODEL_PATH = Path("artifacts/model.pkl")

# Cache the model so it doesn't reload on every run
@st.cache_resource
def load_model():
    return load(MODEL_PATH)

model = load_model()

st.title("Bank Marketing Deposit Prediction")
st.write("Fill in the customer details below to predict whether they will subscribe to a deposit.")

# Input Form 
st.header("Customer Information")

# Numerical Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance (average yearly)", min_value=0.0, value=1000.0)
day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, value=5)
duration = st.number_input("Call Duration (seconds)", min_value=0, value=300)
campaign = st.number_input("Number of Contacts in Current Campaign", min_value=0, value=1)
pdays = st.number_input("Days Passed After Previous Campaign", min_value=-1, max_value=999, value=-1)
previous = st.number_input("Number of Contacts in Previous Campaign", min_value=0, value=0)

# Categorical Inputs
job = st.selectbox("Job", 
                   options=["admin.", "blue-collar", "entrepreneur", "housemaid", "management", 
                            "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
marital = st.selectbox("Marital Status", options=["divorced", "married", "single"])
education = st.selectbox("Education", options=["primary", "secondary", "tertiary", "unknown"])
default_val = st.selectbox("Credit in Default?", options=["no", "yes"])
housing = st.selectbox("Housing Loan", options=["no", "yes"])
loan = st.selectbox("Personal Loan", options=["no", "yes"])
contact = st.selectbox("Contact Type", options=["cellular", "telephone", "unknown"])
month = st.selectbox("Last Contact Month", 
                     options=["apr", "aug", "dec", "feb", "jan", "jul", "jun", "mar", "may", "nov", "oct", "sep"])
poutcome = st.selectbox("Outcome of Previous Campaign", options=["failure", "other", "success", "unknown"])

# Create a DataFrame from the inputs
input_dict = {
    "age": [age],
    "balance": [balance],
    "day": [day],
    "duration": [duration],
    "campaign": [campaign],
    "pdays": [pdays],
    "previous": [previous],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "default": [default_val],
    "housing": [housing],
    "loan": [loan],
    "contact": [contact],
    "month": [month],
    "poutcome": [poutcome],
}
input_df = pd.DataFrame(input_dict)

st.write("### Input Summary")
st.write(input_df)

# Data Preprocessing (One-Hot Encoding)
# List of categorical columns
categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]

# One-hot encode the categorical variables
input_encoded = pd.get_dummies(input_df[categorical_columns], prefix=categorical_columns)

# Get the numerical columns as they are
numerical_columns = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
input_numerical = input_df[numerical_columns]

# Combine the numerical and one-hot encoded categorical features
input_transformed = pd.concat([input_numerical, input_encoded], axis=1)

# Reindex the DataFrame to ensure it has the same columns as expected by the model
input_transformed = input_transformed.reindex(columns=expected_features, fill_value=0)

st.write("### Transformed Input for Prediction")
st.write(input_transformed)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_transformed)[0]
    
    if prediction == 1:
        st.success("Prediction: This client is LIKELY to invest Long Term.")
    else:
        st.warning("Prediction: This customer is UNLIKELY to invest Long Term.")
    
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_transformed)[0][1]
        st.info(f"Probability of subscription: {probability:.2f}")
