from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import joblib
import os

TARGET_COL = "Class"
AMOUNT_COL = "Amount"
V_COLS = [f"V{i}" for i in range(1, 29)]
NUMERIC_COLS = V_COLS + [AMOUNT_COL]

@dataclass
class DataBundle:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    scaler: StandardScaler

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL '{TARGET_COL}' not in data columns: {df.columns.tolist()}")
    missing = [c for c in NUMERIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Edit NUMERIC_COLS in data.py to match your file.")
    df = df[NUMERIC_COLS + [TARGET_COL]].dropna()
    return df

def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> DataBundle:
    X = df[NUMERIC_COLS].copy()
    y = df[TARGET_COL].astype(int).copy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[AMOUNT_COL] = scaler.fit_transform(X_train[[AMOUNT_COL]])
    X_val_scaled[AMOUNT_COL] = scaler.transform(X_val[[AMOUNT_COL]])
    X_test_scaled[AMOUNT_COL] = scaler.transform(X_test[[AMOUNT_COL]])

    return DataBundle(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler)

def save_scaler(scaler: StandardScaler, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
