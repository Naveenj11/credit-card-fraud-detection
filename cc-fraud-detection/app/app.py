import streamlit as st
import pandas as pd
import joblib
import os
from src.data import NUMERIC_COLS, AMOUNT_COL
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")

st.title("💳 Credit Card Fraud Detector")
st.write("Upload a CSV with the same columns as your training data (V1..V28, Amount). We'll predict fraud probability.")

uploaded = st.file_uploader("Upload CSV (no header accepted if columns match exactly)", type=["csv"])

model_path = os.path.join("models", "model.joblib")
scaler_path = os.path.join("models", "scaler.joblib")

if not os.path.exists(model_path):
    st.warning("No trained model found. Please run `python src/train.py --data-path data/creditcard.csv`.")
else:
    model = joblib.load(model_path)
    scaler: StandardScaler = joblib.load(scaler_path)

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in NUMERIC_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            X = df[NUMERIC_COLS].copy()
            # scale Amount
            X[AMOUNT_COL] = scaler.transform(X[[AMOUNT_COL]])
            probs = model.predict_proba(X)[:, 1]
            out = df.copy()
            out['fraud_probability'] = probs
            st.dataframe(out.head(50))
            st.download_button("Download predictions CSV", out.to_csv(index=False), "predictions.csv")
