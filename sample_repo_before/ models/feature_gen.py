import pandas as pd
import numpy as np

def build_features():
    raw = pd.read_csv("data/raw_events.csv")
    raw["amount_capped"] = raw["amount"].clip(0, 5000)
    raw["is_weekend"] = pd.to_datetime(raw["ts"]).dt.dayofweek >= 5
    raw["rolling_7d"] = raw.groupby("user_id")["amount"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    raw["z_score"] = (raw["amount"] - raw["amount"].mean()) / (raw["amount"].std() + 1e-6)
    raw = raw[raw["amount_capped"] > 0.5]
    raw.to_csv("data/features.csv", index=False)
    return raw

if __name__ == "__main__":
    build_features()
