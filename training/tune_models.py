"""Hyperparameter tuning for top 3 algorithms per position.

Usage: python training/tune_models.py
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from training.eval_framework import (
    POSITION_MAP,
    compare_to_baseline,
    load_train_test_unseen,
)
from training.benchmark_models import build_features, _prepare_features

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed — skipping XGBoost.")

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("WARNING: lightgbm not installed — skipping LightGBM.")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Algorithms that handle NaN natively
NATIVE_NAN_ALGORITHMS = {"HistGradientBoosting", "XGBoost", "LightGBM"}

# ── Parameter grids ──

PARAM_GRIDS = {
    "RandomForest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 15, 20, 30],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", 0.5, 0.75, None],
    },
    "HistGradientBoosting": {
        "max_iter": [200, 300, 500],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_leaf": [5, 10, 20, 50],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_leaf_nodes": [15, 31, 63, None],
        "l2_regularization": [0.0, 0.1, 1.0, 10.0],
    },
    "XGBoost": {
        "n_estimators": [200, 300, 500],
        "max_depth": [2, 4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    },
    "LightGBM": {
        "n_estimators": [200, 300, 500],
        "max_depth": [-1, 5, 10, 15],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
        "verbose": [-1],
    },
    "Ridge": {
        "alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    },
    "ElasticNet": {
        "alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "l1_ratio": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95],
        "max_iter": [10000],
    },
}

# ── Base estimators ──

def _get_base_estimator(algo_name: str):
    if algo_name == "RandomForest":
        return RandomForestRegressor(random_state=42)
    elif algo_name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=42)
    elif algo_name == "XGBoost":
        if not HAS_XGBOOST:
            return None
        return XGBRegressor(random_state=42)
    elif algo_name == "LightGBM":
        if not HAS_LIGHTGBM:
            return None
        return LGBMRegressor(random_state=42, verbose=-1)
    elif algo_name == "Ridge":
        return Ridge()
    elif algo_name == "ElasticNet":
        return ElasticNet(max_iter=10000)
    return None

# ── Top 3 algorithms per position (from benchmark results) ──

POSITION_ALGORITHMS = {
    "GK": ["RandomForest", "ElasticNet", "XGBoost"],
    "DEF": ["ElasticNet", "Ridge", "RandomForest"],
    "MID": ["Ridge", "ElasticNet", "HistGradientBoosting"],
    "FWD": ["LightGBM", "RandomForest", "ElasticNet"],
}

POS_NAME_TO_CODE = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}


def tune_models() -> None:
    print("Loading data splits...")
    train_df, test_df, unseen_df = load_train_test_unseen()

    feature_cols = build_features(train_df)
    build_features(test_df)
    build_features(unseen_df)
    print(f"Features: {len(feature_cols)} columns\n")

    all_results = []

    for pos_name in ["GK", "DEF", "MID", "FWD"]:
        pos_code = POS_NAME_TO_CODE[pos_name]
        algos = POSITION_ALGORITHMS[pos_name]

        train_pos = train_df[train_df["element_type"] == pos_code]
        test_pos = test_df[test_df["element_type"] == pos_code]
        unseen_pos = unseen_df[unseen_df["element_type"] == pos_code]

        y_train = train_pos["total_points"]
        y_test = test_pos["total_points"]
        y_unseen = unseen_pos["total_points"]

        print(f"{'=' * 70}")
        print(f"POSITION: {pos_name} (train={len(train_pos)}, test={len(test_pos)}, unseen={len(unseen_pos)})")
        print(f"{'=' * 70}")

        for algo_name in algos:
            estimator = _get_base_estimator(algo_name)
            if estimator is None:
                print(f"  {algo_name}: not available, skipping.")
                continue

            fill_nan = algo_name not in NATIVE_NAN_ALGORITHMS
            X_train = _prepare_features(train_pos, feature_cols, fill_nan)
            X_test = _prepare_features(test_pos, feature_cols, fill_nan)
            X_unseen = _prepare_features(unseen_pos, feature_cols, fill_nan)

            param_grid = PARAM_GRIDS[algo_name]
            n_iter = 30 if algo_name in ("Ridge", "ElasticNet") else 50

            print(f"\n  Tuning {algo_name} (n_iter={n_iter}, 3-fold CV)...", flush=True)
            start = time.time()

            search = RandomizedSearchCV(
                estimator,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=3,
                scoring="neg_mean_absolute_error",
                random_state=42,
                n_jobs=-1,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            elapsed = time.time() - start

            best_model = search.best_estimator_
            best_params = search.best_params_
            cv_mae = -search.best_score_

            # Evaluate on test and unseen
            test_pred = best_model.predict(X_test)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
            test_r2 = r2_score(y_test, test_pred)

            unseen_pred = best_model.predict(X_unseen)
            unseen_mae = mean_absolute_error(y_unseen, unseen_pred)
            unseen_rmse = float(np.sqrt(mean_squared_error(y_unseen, unseen_pred)))
            unseen_r2 = r2_score(y_unseen, unseen_pred)

            print(f"  Done in {elapsed:.1f}s")
            print(f"  Best params: {best_params}")
            print(f"  CV MAE: {cv_mae:.2f} | Test MAE: {test_mae:.2f} | Unseen MAE: {unseen_mae:.2f}")

            all_results.append({
                "position": pos_name,
                "algorithm": algo_name,
                "best_params": str(best_params),
                "cv_mae": cv_mae,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "unseen_mae": unseen_mae,
                "unseen_rmse": unseen_rmse,
                "unseen_r2": unseen_r2,
                "_model": best_model,
            })

    results_df = pd.DataFrame(all_results)

    # ── Summary table ──
    print(f"\n{'=' * 100}")
    print("TUNING RESULTS: All 12 combinations")
    print(f"{'=' * 100}")
    print(f"{'Position':<6} {'Algorithm':<25} {'CV MAE':>8} {'Test MAE':>9} {'Unseen MAE':>11} {'Unseen RMSE':>12} {'Unseen R²':>10}")
    print("-" * 85)
    for _, row in results_df.iterrows():
        print(f"{row['position']:<6} {row['algorithm']:<25} {row['cv_mae']:>8.2f} "
              f"{row['test_mae']:>9.2f} {row['unseen_mae']:>11.2f} "
              f"{row['unseen_rmse']:>12.2f} {row['unseen_r2']:>10.3f}")

    # ── Pick winners per position ──
    print(f"\n{'=' * 80}")
    print("WINNERS per position (lowest unseen MAE)")
    print(f"{'=' * 80}")

    baseline_df = pd.read_csv(BASE_DIR / "training" / "baseline_metrics.csv")

    winners = {}
    for pos_name in ["GK", "DEF", "MID", "FWD"]:
        pos_results = results_df[results_df["position"] == pos_name]
        best_idx = pos_results["unseen_mae"].idxmin()
        best = pos_results.loc[best_idx]

        baseline_row = baseline_df[
            (baseline_df["position"] == pos_name) & (baseline_df["split"] == "unseen")
        ]
        baseline_mae = baseline_row.iloc[0]["mae"] if not baseline_row.empty else float("nan")
        improvement = baseline_mae - best["unseen_mae"]

        print(f"  {pos_name:<5} {best['algorithm']:<25} Unseen MAE: {best['unseen_mae']:.2f}  "
              f"(baseline: {baseline_mae:.2f}, improvement: {improvement:+.2f})")

        winners[pos_name] = best

    # ── Save winning models ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving winning models to {MODEL_DIR}/")
    for pos_name, best in winners.items():
        pos_code = POS_NAME_TO_CODE[pos_name]
        model_path = MODEL_DIR / f"position_model_{pos_code}.joblib"
        joblib.dump(best["_model"], model_path)
        print(f"  {pos_name} ({best['algorithm']}) → {model_path.name}")

    # ── Save CSV ──
    csv_cols = ["position", "algorithm", "best_params", "cv_mae",
                "test_mae", "test_rmse", "test_r2",
                "unseen_mae", "unseen_rmse", "unseen_r2"]
    csv_df = results_df[csv_cols]
    csv_path = BASE_DIR / "training" / "tuning_results.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ── Baseline comparison ──
    winner_rows = []
    for pos_name, best in winners.items():
        winner_rows.append({
            "position": pos_name,
            "mae": best["unseen_mae"],
            "rmse": best["unseen_rmse"],
            "r2": best["unseen_r2"],
        })
    winner_results_df = pd.DataFrame(winner_rows)

    print(f"\n{'=' * 80}")
    print("BASELINE COMPARISON (unseen split)")
    print(f"{'=' * 80}")
    compare_to_baseline(winner_results_df)


if __name__ == "__main__":
    tune_models()
