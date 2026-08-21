import pandas as pd
import joblib
import numpy as np

def score():
    df = pd.read_csv("data/features.csv")
    model = joblib.load("models/model_v3_final_FINAL.pkl")
    scaler = joblib.load("models/scaler_v3.pkl")

    X = df[["amount_capped", "is_weekend", "rolling_7d", "z_score"]].fillna(0)
    X_scaled = scaler.transform(X)
    df["score"] = model.predict_proba(X_scaled)[:, 1]
    df["flag"] = df["score"] > 0.63
    df.to_csv("data/scores_output.csv", index=False)

if __name__ == "__main__":
    score()
