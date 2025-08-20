from __future__ import annotations
import argparse, os, json, joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from data import load_data, split_and_scale

def evaluate(data_path: str, model_dir: str):
    df = load_data(data_path)
    bundle = split_and_scale(df)

    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    y_prob = model.predict_proba(bundle.X_test)[:, 1]

    from utils import compute_core_metrics, best_threshold_by_f1
    best_t, best_f1 = best_threshold_by_f1(bundle.y_test, y_prob)
    metrics = compute_core_metrics(bundle.y_test, y_prob, threshold=best_t)

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics.json:", metrics)

    fig_roc = RocCurveDisplay.from_predictions(bundle.y_test, y_prob)
    plt.title("ROC Curve")
    plt.savefig(os.path.join(model_dir, "roc_curve.png"), bbox_inches="tight")
    plt.close()

    fig_pr = PrecisionRecallDisplay.from_predictions(bundle.y_test, y_prob)
    plt.title("Precision-Recall Curve")
    plt.savefig(os.path.join(model_dir, "pr_curve.png"), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    evaluate(args.data_path, args.model_dir)
