import pandas as pd
import config

_FEATURES_PATH = "data/features.csv"

def build_features():
    raw = pd.read_csv(config.RAW_EVENTS_PATH)
    raw["amount_capped"] = raw["amount"].clip(0, config.AMOUNT_CAP)
    raw["is_weekend"] = pd.to_datetime(raw["ts"]).dt.dayofweek >= 5
    # Refactor slip: dropped min_periods=1 while "simplifying" this line.
    # Looks harmless - it's the same rolling(7) window - but now the first
    # 6 rows per user come back as NaN instead of a partial-window mean.
    raw["rolling_7d"] = raw.groupby("user_id")["amount"].transform(
        lambda s: s.rolling(config.ROLLING_WINDOW_DAYS).mean()
    )
    raw["z_score"] = (raw["amount"] - raw["amount"].mean()) / (
        raw["amount"].std() + config.Z_SCORE_EPSILON
    )
    raw = raw[raw["amount_capped"] > 0.5]
    raw.to_csv(_FEATURES_PATH, index=False)
    return raw

def get_features():
    return pd.read_csv(_FEATURES_PATH)
