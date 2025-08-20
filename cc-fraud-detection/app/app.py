# ------------------- Auto-install missing packages -------------------
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure required packages are installed
required_packages = ["joblib", "pandas", "numpy", "scikit-learn", "streamlit", "matplotlib", "seaborn"]
for pkg in required_packages:
    try:
        __import__(pkg)
    except ModuleNotFoundError:
        install(pkg)
        __import__(pkg)
# ---------------------------------------------------------------------

# Now import packages safely
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# ------------------- App title -------------------
st.title("Credit Card Fraud Detection App")

# ------------------- Upload dataset -------------------
st.header("Upload your dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset preview:")
    st.dataframe(df.head())
    
    # Optional: Show basic stats
    st.write("Dataset summary:")
    st.write(df.describe())

    # ------------------- Model prediction -------------------
    st.header("Make Predictions")
    
    # Load pre-trained model
    try:
        model = joblib.load("model.pkl")  # Make sure model.pkl exists in your repo
    except Exception as e:
        st.error(f"Failed to load model: {e}")
    
    # Example: Let user input feature values manually
    st.subheader("Manual Input")
    if st.button("Predict Example"):
        # Example input, replace with your feature names
        example_input = np.array([[1000, 2, 1]])  # Replace with real feature values
        prediction = model.predict(example_input)
        st.write(f"Prediction result: {prediction[0]}")

# ------------------- Visualization (optional) -------------------
st.header("Exploratory Data Analysis")
if uploaded_file is not None:
    st.subheader("Feature Correlation Heatmap")
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)
