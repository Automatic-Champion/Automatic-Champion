"""Benchmark 6 algorithms × 4 positions on expanded features.

Usage: python training/benchmark_models.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from training.eval_framework import (
    POSITION_MAP,
    compare_to_baseline,
    evaluate_all_positions,
    load_train_test_unseen,
)
from src.team_builder import _add_momentum_features

# Optional imports — skip algorithm if missing
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

# Algorithms that handle NaN natively — do NOT fillna for these
NATIVE_NAN_ALGORITHMS = {"HistGradientBoosting", "XGBoost", "LightGBM"}


def _get_algorithms() -> list[tuple[str, object]]:
    """Return list of (name, model_instance) for all available algorithms."""
    algos = [
        ("RandomForest", RandomForestRegressor(n_estimators=300, random_state=42)),
        ("HistGradientBoosting", HistGradientBoostingRegressor(max_iter=300, random_state=42)),
    ]
    if HAS_XGBOOST:
        algos.append(("XGBoost", XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=42)))
    if HAS_LIGHTGBM:
        algos.append(("LightGBM", LGBMRegressor(n_estimators=300, learning_rate=0.1, random_state=42, verbose=-1)))
    algos.extend([
        ("Ridge", Ridge(alpha=1.0)),
        ("ElasticNet", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)),
    ])
    return algos


def build_features(df: pd.DataFrame) -> list[str]:
    """Build expanded feature columns + momentum features from a DataFrame.

    Returns feature_cols list. Replicates logic from
    src/team_builder._build_feature_cols + _add_momentum_features.
    """
    _add_momentum_features(df)

    features = ["price_now"]
    for col in df.columns:
        if any(col.startswith(f"{y}_years_past_") for y in (1, 2, 3)):
            features.append(col)
        elif col.startswith("momentum_"):
            features.append(col)
    features = [col for col in features if col in df.columns]
    numeric_cols = df[features].select_dtypes(include="number").columns.tolist()
    return numeric_cols


def _prepare_features(df: pd.DataFrame, feature_cols: list[str], fill_nan: bool) -> pd.DataFrame:
    """Extract feature matrix, optionally filling NaN with 0."""
    X = df[feature_cols].copy()
    if fill_nan:
        X = X.fillna(0)
    return X


def train_and_evaluate() -> None:
    print("Loading data splits...")
    train_df, test_df, unseen_df = load_train_test_unseen()

    # Build features (adds momentum columns in-place)
    feature_cols = build_features(train_df)
    build_features(test_df)
    build_features(unseen_df)

    print(f"Feature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")

    algorithms = _get_algorithms()
    all_results = []

    for pos_code, pos_name in POSITION_MAP.items():
        train_pos = train_df[train_df["element_type"] == pos_code]
        test_pos = test_df[test_df["element_type"] == pos_code]
        unseen_pos = unseen_df[unseen_df["element_type"] == pos_code]

        if train_pos.empty:
            print(f"  {pos_name}: no training data, skipping.")
            continue

        y_train = train_pos["total_points"]
        y_test = test_pos["total_points"]
        y_unseen = unseen_pos["total_points"]

        for algo_name, model_template in algorithms:
            fill_nan = algo_name not in NATIVE_NAN_ALGORITHMS

            X_train = _prepare_features(train_pos, feature_cols, fill_nan)
            X_test = _prepare_features(test_pos, feature_cols, fill_nan)
            X_unseen = _prepare_features(unseen_pos, feature_cols, fill_nan)

            # Clone the model (sklearn clone semantics via fresh construction)
            from sklearn.base import clone
            model = clone(model_template)

            model.fit(X_train, y_train)

            # Test metrics
            test_pred = model.predict(X_test)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
            test_r2 = r2_score(y_test, test_pred)

            # Unseen metrics
            unseen_pred = model.predict(X_unseen)
            unseen_mae = mean_absolute_error(y_unseen, unseen_pred)
            unseen_rmse = float(np.sqrt(mean_squared_error(y_unseen, unseen_pred)))
            unseen_r2 = r2_score(y_unseen, unseen_pred)

            all_results.append({
                "position": pos_name,
                "algorithm": algo_name,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "unseen_mae": unseen_mae,
                "unseen_rmse": unseen_rmse,
                "unseen_r2": unseen_r2,
                "_model": model,
                "_pos_code": pos_code,
                "_fill_nan": fill_nan,
            })

            print(f"  {pos_name:>3} | {algo_name:<25} | unseen MAE: {unseen_mae:.2f}")

    results_df = pd.DataFrame(all_results)

    # ── Table 1: Full benchmark ──
    print("\n" + "=" * 90)
    print("TABLE 1: Full Benchmark (all algorithms × all positions)")
    print("=" * 90)
    print(f"{'Position':<10} {'Algorithm':<25} {'Test MAE':>9} {'Test RMSE':>10} "
          f"{'Unseen MAE':>11} {'Unseen RMSE':>12} {'Unseen R²':>10}")
    print("-" * 90)
    for _, row in results_df.sort_values(["position", "unseen_mae"]).iterrows():
        print(f"{row['position']:<10} {row['algorithm']:<25} {row['test_mae']:>9.2f} "
              f"{row['test_rmse']:>10.2f} {row['unseen_mae']:>11.2f} "
              f"{row['unseen_rmse']:>12.2f} {row['unseen_r2']:>10.3f}")

    # ── Table 2: Winners per position ──
    baseline_df = pd.read_csv(BASE_DIR / "training" / "baseline_metrics.csv")

    print("\n" + "=" * 80)
    print("TABLE 2: Winners per position (lowest unseen MAE)")
    print("=" * 80)
    print(f"{'Position':<10} {'Best Algorithm':<25} {'Unseen MAE':>11} "
          f"{'Baseline':>9} {'Improvement':>12}")
    print("-" * 80)

    winners = {}
    for pos_name in ["GK", "DEF", "MID", "FWD"]:
        pos_results = results_df[results_df["position"] == pos_name]
        best = pos_results.loc[pos_results["unseen_mae"].idxmin()]

        baseline_row = baseline_df[
            (baseline_df["position"] == pos_name) & (baseline_df["split"] == "unseen")
        ]
        baseline_mae = baseline_row.iloc[0]["mae"] if not baseline_row.empty else float("nan")
        improvement = baseline_mae - best["unseen_mae"]

        print(f"{pos_name:<10} {best['algorithm']:<25} {best['unseen_mae']:>11.2f} "
              f"{baseline_mae:>9.2f} {improvement:>+11.2f}")

        winners[pos_name] = best

    # ── Save winning models ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pos_name_to_code = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}

    for pos_name, best in winners.items():
        pos_code = pos_name_to_code[pos_name]
        model_path = MODEL_DIR / f"position_model_{pos_code}.joblib"
        joblib.dump(best["_model"], model_path)
        print(f"  Saved {pos_name} winner ({best['algorithm']}) → {model_path.name}")

    # ── Table 3: Baseline comparison via compare_to_baseline ──
    # Build unseen results in the format compare_to_baseline expects
    winner_rows = []
    for pos_name, best in winners.items():
        winner_rows.append({
            "position": pos_name,
            "mae": best["unseen_mae"],
            "rmse": best["unseen_rmse"],
            "r2": best["unseen_r2"],
        })
    winner_results_df = pd.DataFrame(winner_rows)

    print("\n" + "=" * 80)
    print("TABLE 3: Baseline comparison (unseen split)")
    print("=" * 80)
    compare_to_baseline(winner_results_df)

    # ── Save CSV ──
    csv_cols = ["position", "algorithm", "test_mae", "test_rmse", "test_r2",
                "unseen_mae", "unseen_rmse", "unseen_r2"]
    csv_df = results_df[csv_cols]
    csv_path = BASE_DIR / "training" / "benchmark_results.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(f"Models saved to {MODEL_DIR}")


if __name__ == "__main__":
    train_and_evaluate()
