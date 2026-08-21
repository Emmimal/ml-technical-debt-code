import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
import feature_store


def main():
    df = feature_store.get_features()
    X = df[config.FEATURE_COLUMNS].fillna(0)
    y = df[config.LABEL_COLUMN]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=config.TEST_SIZE, random_state=config.TRAIN_TEST_SPLIT_SEED
    )

    model = LogisticRegression(
        C=config.LOGREG_C, max_iter=config.LOGREG_MAX_ITER, class_weight=config.CLASS_WEIGHT
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print("acc", acc)

    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)


if __name__ == "__main__":
    main()
