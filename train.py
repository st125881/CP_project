"""Rebuild the tuned Random Forest and save it for deployment.

Usage:  python train.py battery_total_cycle_life_full.csv
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features import FEATURES, battery_features, first_50_cycles

# Best parameters found by RandomizedSearchCV in the tuning notebook
BEST_PARAMS = dict(
    n_estimators=400,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features=0.8,
    max_depth=None,
)


def build():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1, **BEST_PARAMS)),
    ])


def main(csv_path):
    df = pd.read_csv(csv_path)
    first50 = first_50_cycles(df)
    X = battery_features(first50)
    y = first50.groupby("Battery")["total_cycle_life"].first()
    ok = y.notna()
    X, y = X[ok], y[ok]
    print(f"Batteries: {len(X)}")

    # 1) Re-check against the notebook (same split, seed 42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
    check = build().fit(X_tr, y_tr)
    pred = check.predict(X_te)
    metrics = {
        "MAE": float(mean_absolute_error(y_te, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
        "R2": float(r2_score(y_te, pred)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    }
    print("Held-out test (should be close to MAE 151.24 / RMSE 240.78 / R2 0.6063):")
    print(metrics)

    # 2) Final model: same hyperparameters, refit on all batteries
    final = build().fit(X, y)
    joblib.dump(final, "models/tuned_rf.joblib")

    meta = {
        "features": FEATURES,
        "best_params": BEST_PARAMS,
        "heldout_metrics": metrics,
        "n_batteries_final_fit": int(len(X)),
        "sklearn_version": sklearn.__version__,
        "feature_ranges": {
            f: [float(X[f].min()), float(X[f].max()), float(X[f].median())]
            for f in FEATURES
        },
        "target_range": [float(y.min()), float(y.max())],
    }
    json.dump(meta, open("models/model_meta.json", "w"), indent=2)
    print("Saved models/tuned_rf.joblib and models/model_meta.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "battery_total_cycle_life_full.csv")
