import joblib
import config
import feature_store


def score():
    df = feature_store.get_features()
    model = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)

    X = df[config.FEATURE_COLUMNS].fillna(0)
    X_scaled = scaler.transform(X)
    df["score"] = model.predict_proba(X_scaled)[:, 1]
    df["flag"] = df["score"] > config.SCORE_THRESHOLD
    df.to_csv("data/scores_output.csv", index=False)


if __name__ == "__main__":
    score()
