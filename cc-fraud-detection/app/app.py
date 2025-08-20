import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/model.pkl")

st.title("💳 Credit Card Fraud Detection")

st.write("Upload transaction data (CSV) to check for fraud.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview:", data.head())

    preds = model.predict(data)
    data['Fraud_Prediction'] = preds
    st.write("### Results:", data.head())

    fraud_count = data['Fraud_Prediction'].sum()
    st.metric("Fraudulent Transactions", fraud_count)
