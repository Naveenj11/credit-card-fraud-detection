from __future__ import annotations
import argparse, os, json, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from data import load_data, split_and_scale, save_scaler

def train_models(data_path: str, model_dir: str):
    df = load_data(data_path)
    bundle = split_and_scale(df)

    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(bundle.X_train, bundle.y_train)

    rf = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, class_weight="balanced_subsample", random_state=42
    )
    rf.fit(bundle.X_train, bundle.y_train)

    lr_val = roc_auc_score(bundle.y_val, lr.predict_proba(bundle.X_val)[:,1])
    rf_val = roc_auc_score(bundle.y_val, rf.predict_proba(bundle.X_val)[:,1])

    best_model, best_name, best_score = (lr, "logreg", lr_val) if lr_val >= rf_val else (rf, "random_forest", rf_val)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "model_meta.json"), "w") as f:
        json.dump({"best_model": best_name, "val_roc_auc": best_score}, f, indent=2)

    save_scaler(bundle.scaler, model_dir)
    print(f"Saved best model: {best_name} ROC-AUC(val)={best_score:.4f} to {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    train_models(args.data_path, args.model_dir)
