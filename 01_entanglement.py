"""
Entanglement (CACE - Changing Anything Changes Everything) demonstration.

Trains a logistic regression on 10 features under two regimes:
  - CORRELATED: features share latent structure (typical of production feature
    sets, where many features are derived from overlapping raw signals)
  - INDEPENDENT: features are statistically independent (a decoupled system)

Then removes a single feature (simulating a routine deprecation - a raw
signal gets retired, a vendor field disappears) and retrains from scratch.
Measures how much the REMAINING features' learned coefficients and the
model's predictions on a fixed holdout set shift, even though nothing about
those remaining features changed on disk.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

SEED = 42
N = 20000
N_FEATURES = 10


def make_data(rng, correlated: bool):
    if correlated:
        # Strong shared latent structure: 2 latent factors drive all 10
        # features, so knowing any 8-9 of them lets a regularized model
        # reconstruct most of what the missing one carried. This is the
        # normal state of an engineered feature set (ratios, rolling
        # aggregates, and counts derived from the same handful of raw events).
        latent = rng.normal(size=(N, 2))
        loadings = rng.uniform(0.7, 1.1, size=(2, N_FEATURES))
        signs = rng.choice([-1, 1], size=(2, N_FEATURES))
        X = latent @ (loadings * signs) + rng.normal(scale=0.35, size=(N, N_FEATURES))
    else:
        X = rng.normal(size=(N, N_FEATURES))

    true_w = rng.uniform(-1.5, 1.5, size=N_FEATURES)
    logits = X @ true_w + rng.normal(scale=0.5, size=N)
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=N) < p).astype(int)
    return X, y


def fit(X, y):
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(X, y)
    return model


def run_condition(correlated: bool):
    rng = np.random.default_rng(SEED)
    X, y = make_data(rng, correlated)

    split = int(N * 0.7)
    X_train, y_train = X[:split], y[:split]
    X_test = X[split:]

    baseline_model = fit(X_train, y_train)
    baseline_coef = baseline_model.coef_[0].copy()
    baseline_pred = baseline_model.predict_proba(X_test)[:, 1]

    coef_shifts = []
    pred_shifts = []
    for dropped_idx in range(N_FEATURES):
        keep_idx = [i for i in range(N_FEATURES) if i != dropped_idx]
        X_train_mod = X_train[:, keep_idx]
        X_test_mod = X_test[:, keep_idx]

        new_model = fit(X_train_mod, y_train)
        new_coef = new_model.coef_[0].copy()
        new_pred = new_model.predict_proba(X_test_mod)[:, 1]

        baseline_kept_coef = baseline_coef[keep_idx]
        rel_coef_shift = np.mean(
            np.abs(new_coef - baseline_kept_coef) / (np.abs(baseline_kept_coef) + 1e-8)
        )
        pred_shift = np.mean(np.abs(new_pred - baseline_pred))

        coef_shifts.append(rel_coef_shift)
        pred_shifts.append(pred_shift)

    return np.mean(coef_shifts), np.mean(pred_shifts)


if __name__ == "__main__":
    corr_coef_shift, corr_pred_shift = run_condition(correlated=True)
    indep_coef_shift, indep_pred_shift = run_condition(correlated=False)

    print("=== Entanglement (CACE) results, drop one feature, retrain ===")
    print(f"Correlated features : mean |D coef| on remaining features = {corr_coef_shift*100:.1f}%, "
          f"mean |D predicted prob| = {corr_pred_shift:.4f}")
    print(f"Independent features: mean |D coef| on remaining features = {indep_coef_shift*100:.1f}%, "
          f"mean |D predicted prob| = {indep_pred_shift:.4f}")
    print(f"Coefficient-shift ratio (correlated / independent): {corr_coef_shift/indep_coef_shift:.2f}x")
    print(f"Prediction-shift ratio (correlated / independent): {corr_pred_shift/indep_pred_shift:.2f}x")
