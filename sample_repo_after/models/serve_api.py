from flask import Flask, request, jsonify
import joblib
import config
import feature_store

app = Flask(__name__)
model = joblib.load(config.MODEL_PATH)
scaler = joblib.load(config.SCALER_PATH)
_lookup_cache = feature_store.get_features().set_index("user_id")


@app.route("/score", methods=["POST"])
def score():
    user_id = request.json["user_id"]
    row = _lookup_cache.loc[user_id]
    X = [[row[c] for c in config.FEATURE_COLUMNS]]
    X_scaled = scaler.transform(X)
    p = model.predict_proba(X_scaled)[0, 1]
    return jsonify({"score": float(p), "flag": bool(p > config.SCORE_THRESHOLD)})


if __name__ == "__main__":
    app.run(port=5000)
