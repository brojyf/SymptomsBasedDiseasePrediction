import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

def load_model(model_path: str, input_size: int, hidden_layers: list, num_classes: int, device: str):
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

def get_topk(probs: np.ndarray, encoder: "LabelEncoder", k: int = 5):
    idx = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
    labels = [[encoder.inverse_transform([i])[0] for i in row] for row in idx]
    probs_k = np.sort(probs, axis=1)[:, -k:][:, ::-1]
    return labels, probs_k

def main():
    MODEL_PATH = "output/mlp_disease_model.pth"
    ENCODER_PATH = "output/label_encoder.pkl"
    SCALER_PATH = "output/scaler.pkl"

    p = argparse.ArgumentParser()
    p.add_argument("--input",       required=True)
    p.add_argument("--output",      required=True)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--hidden",      nargs="+", type=int, default=[256,128])
    args = p.parse_args()

    # Read CSV
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    symptom_cols = df.columns.tolist()
    X = df[symptom_cols].values.astype(float)

    # Preprocess and model
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    Xs = scaler.transform(X)

    model, dev = load_model(
        MODEL_PATH,
        input_size=Xs.shape[1],
        hidden_layers=args.hidden,
        num_classes=len(encoder.classes_),
        device=args.device
    )

    # Predict
    with torch.no_grad():
        t = torch.from_numpy(Xs).float().to(dev)
        logits = model(t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    # Top 5
    labels_k, probs_k = get_topk(probs, encoder, k=5)

    # Concat
    disease_list = []
    symptoms_list = []
    for i, row in df.iterrows():
        items = [
            f"{labels_k[i][j]}({probs_k[i][j]:.4f})"
            for j in range(5)
        ]
        disease_list.append(";".join(items))

        present = [col for col in symptom_cols if row[col] == 1]
        symptoms_list.append(",".join(present))

    out = pd.DataFrame({
        "disease_top5": disease_list,
        "symptoms": symptoms_list
    })

    # Write
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print("Save to: ", args.output)


if __name__ == "__main__":
    main()