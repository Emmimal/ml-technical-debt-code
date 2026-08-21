import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    df = pd.read_csv("data/features.csv")
    X = df[["amount_capped", "is_weekend", "rolling_7d", "z_score"]].fillna(0)
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=1234
    )

    model = LogisticRegression(C=0.37, max_iter=500, class_weight={0: 1, 1: 4.2})
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print("acc", acc)

    joblib.dump(model, "models/model_v3_final_FINAL.pkl")
    joblib.dump(scaler, "models/scaler_v3.pkl")

if __name__ == "__main__":
    main()
