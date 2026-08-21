import pandas as pd
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("models/model_v3_final_FINAL.pkl")
scaler = joblib.load("models/scaler_v3.pkl")
_lookup_cache = pd.read_csv("data/features.csv").set_index("user_id")

@app.route("/score", methods=["POST"])
def score():
    user_id = request.json["user_id"]
    row = _lookup_cache.loc[user_id]
    X = [[row["amount_capped"], row["is_weekend"], row["rolling_7d"], row["z_score"]]]
    X_scaled = scaler.transform(X)
    p = model.predict_proba(X_scaled)[0, 1]
    return jsonify({"score": float(p), "flag": bool(p > 0.63)})

if __name__ == "__main__":
    app.run(port=5000)
