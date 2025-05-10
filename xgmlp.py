import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import subprocess
from xgboost import XGBClassifier


MLP_MODEL_PATH   = "mlp/output/mlp_disease_model.pth"
SCALER_PATH      = "mlp/output/scaler.pkl"
ENCODER_PATH     = "xgb/output/label_encoder.pkl"
XGB_MODEL_PATH   = "xgb/output/xgb_model.json"

def load_mlp(model_path: str, input_size: int, hidden_layers: list, num_classes: int, device: str):
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_layers, num_classes, dropout=0.2):
            super(MLP, self).__init__()
            layers = []
            in_features = input_size

            for hidden_size in hidden_layers:
                layers.append(nn.Linear(in_features, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_features = hidden_size

            layers.append(nn.Linear(in_features, num_classes))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    dev = torch.device(device)
    model = MLP(input_size, hidden_layers, num_classes).to(dev)
    state = torch.load(model_path, map_location=dev)
    model.load_state_dict(state)
    model.eval()
    return model, dev

def get_topk(probs: np.ndarray, encoder, k: int = 5):
    idx = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
    labels = [[encoder.inverse_transform([i])[0] for i in row] for row in idx]
    vals   = np.sort(probs, axis=1)[:, -k:][:, ::-1]
    return labels, vals

def is_gpu_available() -> bool:
    try:
        return subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode == 0
    except FileNotFoundError:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="input csv file")
    parser.add_argument("--output", required=True, help="output csv file")
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument("--hidden", nargs="+", type=int, default=[256,128], help="mlp hidden layers")
    args = parser.parse_args()

    # Read csv
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    symptom_cols = df.columns.tolist()
    X_raw = df[symptom_cols].values.astype(float)

    # Load encoder & scaler
    scaler  = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    Xs = scaler.transform(X_raw)

    # mlp
    mlp_model, dev = load_mlp(
        MLP_MODEL_PATH,
        input_size = Xs.shape[1],
        hidden_layers = args.hidden,
        num_classes = len(encoder.classes_),
        device = args.device
    )
    with torch.no_grad():
        t       = torch.from_numpy(Xs).float().to(dev)
        logits  = mlp_model(t)
        probs_m = torch.softmax(logits, dim=1).cpu().numpy()

    # xgboost
    if is_gpu_available():
        print("Using GPU for XGBoost")
        xgb_params = {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}
    else:
        print("Using CPU for XGBoost")
        xgb_params = {"tree_method": "hist", "predictor": "cpu_predictor"}

    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.load_model(XGB_MODEL_PATH)
    probs_x = xgb_model.predict_proba(X_raw)

    # Top 5
    probs_avg = (probs_m + probs_x) / 2.0
    labels_u, vals_u = get_topk(probs_avg, encoder, k=5)
    unified_top5 = [
        ";".join(f"{labels_u[i][j]}({vals_u[i][j]:.4f})" for j in range(5))
        for i in range(len(df))
    ]

    # Construct col
    symptoms = [
        ",".join([col for col, v in row.items() if v == 1])
        for _, row in df.iterrows()
    ]

    # Output
    out_df = pd.DataFrame({
        "top5_unified": unified_top5,
        "symptoms":      symptoms
    })
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print("Predictions saved to: ", args.output)

if __name__ == "__main__":
    main()
