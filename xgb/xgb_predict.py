import pandas as pd
import numpy as np
import joblib
import subprocess
from xgboost import XGBClassifier

# PATH
CSV_PATH   = "sample.csv"
MODEL_PATH = "output/xgb_model.json"
LE_PATH    = "output/label_encoder.pkl"
OUT_PATH = "xgb_predictions.csv"

def is_gpu_available() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

# Read csv
df_raw = pd.read_csv(CSV_PATH)
symptom_cols = df_raw.columns.tolist()

# Check GPU
if is_gpu_available():
    print("Using GPU")
    xgb_params = {
        "tree_method": "gpu_hist",
        "predictor":   "gpu_predictor",
    }
else:
    print("USING CPU")
    xgb_params = {
        "tree_method": "hist",
        "predictor":   "cpu_predictor",
    }

# Load Model and Predict
model = XGBClassifier()
model.load_model(MODEL_PATH)
y_proba = model.predict_proba(df_raw)

# Load Label encoder
le = joblib.load(LE_PATH)
if hasattr(le, "classes_"):
    class_names = le.classes_
else:
    class_names = np.array(le, dtype=str)

# Get Top 5
top_n    = 5
top_idx  = np.argsort(y_proba, axis=1)[:, ::-1][:, :top_n]
top_vals = np.sort(y_proba,  axis=1)[:, ::-1][:, :top_n]

# Construct Columns
df_raw["top5_prediction"] = [
    ", ".join(
        f"{class_names[idx]}({prob:.3f})"
        for idx, prob in zip(top_idx[i], top_vals[i])
    )
    for i in range(len(df_raw))
]
df_raw["all_symptoms"] = df_raw[symptom_cols].apply(
    lambda row: ", ".join([col for col, v in row.items() if v == 1]),
    axis=1
)

# Final Output
output = df_raw[["top5_prediction", "all_symptoms"]]
output.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("Predictions saved to {}".format(OUT_PATH))