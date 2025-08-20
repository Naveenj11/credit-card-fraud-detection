# Credit Card Fraud Detection (End-to-End)
Detect fraudulent transactions using classic ML with strong evaluation for imbalanced data.

## Dataset
Place the CSV at `data/creditcard.csv` (columns like the popular EU card dataset: `V1..V28`, `Amount`, `Class`). If your schema differs, update `src/data.py`.

## Quickstart
```bash
pip install -r requirements.txt
python src/train.py --data-path data/creditcard.csv --model-dir models
python src/evaluate.py --data-path data/creditcard.csv --model-dir models
streamlit run app/app.py
```

## Notes
- **Preprocessing**: StandardScaler on `Amount` only; `V1..V28` kept as-is.
- **Models**: LogisticRegression (class_weight='balanced') + RandomForest baseline; best by ROC-AUC (val).
- **Imbalance**: class_weight + threshold tuning from PR curve.
- **Metrics**: ROC-AUC, PR-AUC, F1, Confusion Matrix; emphasize PR-AUC for imbalance.
