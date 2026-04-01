"""Feature selection, pruned retraining, and ensemble training.

Usage: python training/train_final_models.py
"""
from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from training.eval_framework import (
    POSITION_MAP,
    compare_to_baseline,
    load_train_test_unseen,
)
from training.benchmark_models import build_features, _prepare_features

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
TRAINING_DIR = BASE_DIR / "training"

POS_NAME_TO_CODE = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}

# Algorithms that handle NaN natively
NATIVE_NAN_ALGORITHMS = {"HistGradientBoosting", "XGBoost", "LightGBM"}

# ── Tuned hyperparameters from Task 4 (parsed from tuning_results.csv) ──


def _load_tuning_results() -> pd.DataFrame:
    path = TRAINING_DIR / "tuning_results.csv"
    return pd.read_csv(path)


def _parse_best_params(params_str: str) -> dict:
    return ast.literal_eval(params_str)


def _build_estimator(algo_name: str, params: dict):
    """Build a fresh estimator with the given hyperparameters."""
    if algo_name == "XGBoost":
        if XGBRegressor is None:
            raise ImportError("xgboost not installed")
        return XGBRegressor(random_state=42, **params)
    elif algo_name == "LightGBM":
        if LGBMRegressor is None:
            raise ImportError("lightgbm not installed")
        params.setdefault("verbose", -1)
        return LGBMRegressor(random_state=42, **params)
    elif algo_name == "RandomForest":
        return RandomForestRegressor(random_state=42, **params)
    elif algo_name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=42, **params)
    elif algo_name == "Ridge":
        return Ridge(**params)
    elif algo_name == "ElasticNet":
        return ElasticNet(**params)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def _evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


# ── Winners per position from tuning ──

POSITION_WINNERS = {
    "GK": "XGBoost",
    "DEF": "ElasticNet",
    "MID": "Ridge",
    "FWD": "LightGBM",
}

# ── Ensemble composition (top 3 per position from tuning) ──

ENSEMBLE_MEMBERS = {
    "GK": ["XGBoost", "RandomForest", "ElasticNet"],
    "DEF": ["ElasticNet", "Ridge", "RandomForest"],
    "MID": ["Ridge", "ElasticNet", "HistGradientBoosting"],
    "FWD": ["LightGBM", "ElasticNet", "RandomForest"],
}


def _get_winner_params(tuning_df: pd.DataFrame, pos_name: str, algo_name: str) -> dict:
    row = tuning_df[
        (tuning_df["position"] == pos_name) & (tuning_df["algorithm"] == algo_name)
    ]
    if row.empty:
        raise ValueError(f"No tuning result for {pos_name}/{algo_name}")
    return _parse_best_params(row.iloc[0]["best_params"])


def _needs_fillna(algo_name: str) -> bool:
    return algo_name not in NATIVE_NAN_ALGORITHMS


def train_final_models() -> None:
    print("=" * 80)
    print("FINAL MODEL TRAINING: Feature Selection + Pruned Retraining + Ensembles")
    print("=" * 80)

    # Load data
    print("\nLoading data splits...")
    train_df, test_df, unseen_df = load_train_test_unseen()

    feature_cols = build_features(train_df)
    build_features(test_df)
    build_features(unseen_df)
    print(f"Total features: {len(feature_cols)}")

    tuning_df = _load_tuning_results()

    selected_features: dict[str, list[str]] = {}
    all_results: list[dict] = []

    for pos_name in ["GK", "DEF", "MID", "FWD"]:
        pos_code = POS_NAME_TO_CODE[pos_name]
        winner_algo = POSITION_WINNERS[pos_name]

        train_pos = train_df[train_df["element_type"] == pos_code]
        test_pos = test_df[test_df["element_type"] == pos_code]
        unseen_pos = unseen_df[unseen_df["element_type"] == pos_code]

        y_train = train_pos["total_points"]
        y_test = test_pos["total_points"]
        y_unseen = unseen_pos["total_points"]

        print(f"\n{'=' * 70}")
        print(f"POSITION: {pos_name} (train={len(train_pos)}, test={len(test_pos)}, unseen={len(unseen_pos)})")
        print(f"{'=' * 70}")

        # ── Part A: Feature selection via permutation importance ──
        print(f"\n  Part A: Feature selection using {winner_algo}...")

        winner_params = _get_winner_params(tuning_df, pos_name, winner_algo)
        fill_nan = _needs_fillna(winner_algo)

        X_train_full = _prepare_features(train_pos, feature_cols, fill_nan)
        X_test_full = _prepare_features(test_pos, feature_cols, fill_nan)
        X_unseen_full = _prepare_features(unseen_pos, feature_cols, fill_nan)

        # Train the winner model on full features for permutation importance
        model_full = _build_estimator(winner_algo, winner_params)
        model_full.fit(X_train_full, y_train)

        # Compute permutation importance on test set
        perm_result = permutation_importance(
            model_full,
            X_test_full,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="neg_mean_absolute_error",
        )

        # Keep features where mean importance > 0
        importances_mean = perm_result.importances_mean
        keep_mask = importances_mean > 0
        kept_features = [f for f, keep in zip(feature_cols, keep_mask) if keep]

        if len(kept_features) == 0:
            print(f"  WARNING: No features with positive importance, keeping all.")
            kept_features = list(feature_cols)

        removed = len(feature_cols) - len(kept_features)
        selected_features[str(pos_code)] = kept_features
        print(f"  Selected {len(kept_features)}/{len(feature_cols)} features (removed {removed})")

        # Show top 10 features
        sorted_idx = np.argsort(importances_mean)[::-1]
        print(f"  Top 10 features:")
        for i in sorted_idx[:10]:
            print(f"    {feature_cols[i]:40s} importance={importances_mean[i]:.4f}")

        # ── Evaluate Task 4 winner (full features) ──
        task4_test = _evaluate(model_full, X_test_full, y_test)
        task4_unseen = _evaluate(model_full, X_unseen_full, y_unseen)
        print(f"\n  Task4 single ({winner_algo}, 74 features):")
        print(f"    Test MAE: {task4_test['mae']:.2f} | Unseen MAE: {task4_unseen['mae']:.2f}")

        all_results.append({
            "position": pos_name,
            "approach": f"Task4 single ({winner_algo})",
            "algorithm": winner_algo,
            "n_features": len(feature_cols),
            "test_mae": task4_test["mae"],
            "test_rmse": task4_test["rmse"],
            "test_r2": task4_test["r2"],
            "unseen_mae": task4_unseen["mae"],
            "unseen_rmse": task4_unseen["rmse"],
            "unseen_r2": task4_unseen["r2"],
            "_model": model_full,
            "_features": feature_cols,
        })

        # ── Part B: Retrain with pruned features ──
        print(f"\n  Part B: Retrain {winner_algo} with {len(kept_features)} pruned features...")

        X_train_pruned = _prepare_features(train_pos, kept_features, fill_nan)
        X_test_pruned = _prepare_features(test_pos, kept_features, fill_nan)
        X_unseen_pruned = _prepare_features(unseen_pos, kept_features, fill_nan)

        model_pruned = _build_estimator(winner_algo, winner_params)
        model_pruned.fit(X_train_pruned, y_train)

        pruned_test = _evaluate(model_pruned, X_test_pruned, y_test)
        pruned_unseen = _evaluate(model_pruned, X_unseen_pruned, y_unseen)
        print(f"    Test MAE: {pruned_test['mae']:.2f} | Unseen MAE: {pruned_unseen['mae']:.2f}")

        all_results.append({
            "position": pos_name,
            "approach": f"Pruned single ({winner_algo})",
            "algorithm": winner_algo,
            "n_features": len(kept_features),
            "test_mae": pruned_test["mae"],
            "test_rmse": pruned_test["rmse"],
            "test_r2": pruned_test["r2"],
            "unseen_mae": pruned_unseen["mae"],
            "unseen_rmse": pruned_unseen["rmse"],
            "unseen_r2": pruned_unseen["r2"],
            "_model": model_pruned,
            "_features": kept_features,
        })

        # ── Part C: Ensemble (VotingRegressor) with pruned features ──
        ensemble_algos = ENSEMBLE_MEMBERS[pos_name]
        print(f"\n  Part C: Ensemble [{', '.join(ensemble_algos)}] with {len(kept_features)} pruned features...")

        # All ensemble sub-models must see the same data → fillna(0) for all
        X_train_ens = _prepare_features(train_pos, kept_features, fill_nan=True)
        X_test_ens = _prepare_features(test_pos, kept_features, fill_nan=True)
        X_unseen_ens = _prepare_features(unseen_pos, kept_features, fill_nan=True)

        estimators = []
        for algo_name in ensemble_algos:
            params = _get_winner_params(tuning_df, pos_name, algo_name)
            est = _build_estimator(algo_name, params)
            estimators.append((algo_name, est))

        ensemble = VotingRegressor(estimators=estimators)
        ensemble.fit(X_train_ens, y_train)

        ens_test = _evaluate(ensemble, X_test_ens, y_test)
        ens_unseen = _evaluate(ensemble, X_unseen_ens, y_unseen)
        print(f"    Test MAE: {ens_test['mae']:.2f} | Unseen MAE: {ens_unseen['mae']:.2f}")

        all_results.append({
            "position": pos_name,
            "approach": f"Pruned ensemble",
            "algorithm": f"Ensemble({'+'.join(ensemble_algos)})",
            "n_features": len(kept_features),
            "test_mae": ens_test["mae"],
            "test_rmse": ens_test["rmse"],
            "test_r2": ens_test["r2"],
            "unseen_mae": ens_unseen["mae"],
            "unseen_rmse": ens_unseen["rmse"],
            "unseen_r2": ens_unseen["r2"],
            "_model": ensemble,
            "_features": kept_features,
        })

    # ── Table 1: Feature selection results ──
    print(f"\n{'=' * 80}")
    print("TABLE 1: Feature Selection Results")
    print(f"{'=' * 80}")
    print(f"{'Position':<10} {'Original Features':>18} {'Selected Features':>18} {'Removed':>10}")
    print("-" * 60)
    for pos_name in ["GK", "DEF", "MID", "FWD"]:
        pos_code = POS_NAME_TO_CODE[pos_name]
        n_selected = len(selected_features[str(pos_code)])
        n_removed = len(feature_cols) - n_selected
        print(f"{pos_name:<10} {len(feature_cols):>18} {n_selected:>18} {n_removed:>10}")

    # ── Table 2: Three-way comparison ──
    results_df = pd.DataFrame(all_results)
    print(f"\n{'=' * 100}")
    print("TABLE 2: Three-Way Comparison per Position")
    print(f"{'=' * 100}")
    print(f"{'Position':<6} {'Approach':<35} {'Features':>8} {'Unseen MAE':>11} {'Unseen RMSE':>12} {'Unseen R²':>10}")
    print("-" * 85)
    for _, row in results_df.iterrows():
        print(f"{row['position']:<6} {row['approach']:<35} {row['n_features']:>8} "
              f"{row['unseen_mae']:>11.2f} {row['unseen_rmse']:>12.2f} {row['unseen_r2']:>10.3f}")

    # ── Pick winners per position (lowest unseen MAE) ──
    print(f"\n{'=' * 80}")
    print("WINNERS per position (lowest unseen MAE)")
    print(f"{'=' * 80}")

    baseline_df = pd.read_csv(TRAINING_DIR / "baseline_metrics.csv")
    winners = {}

    for pos_name in ["GK", "DEF", "MID", "FWD"]:
        pos_results = results_df[results_df["position"] == pos_name]
        best_idx = pos_results["unseen_mae"].idxmin()
        best = pos_results.loc[best_idx]
        winners[pos_name] = best

        baseline_row = baseline_df[
            (baseline_df["position"] == pos_name) & (baseline_df["split"] == "unseen")
        ]
        baseline_mae = baseline_row.iloc[0]["mae"] if not baseline_row.empty else float("nan")
        improvement = baseline_mae - best["unseen_mae"]

        print(f"  {pos_name:<5} {best['approach']:<35} Unseen MAE: {best['unseen_mae']:.2f}  "
              f"(baseline: {baseline_mae:.2f}, improvement: {improvement:+.2f})")

    # ── Table 3: Final winners vs original baseline ──
    print(f"\n{'=' * 80}")
    print("TABLE 3: Final Winners vs Original Baseline")
    print(f"{'=' * 80}")
    print(f"{'Position':<10} {'Final Model':<35} {'Unseen MAE':>11} {'Baseline':>10} {'Improvement':>12}")
    print("-" * 80)
    for pos_name in ["GK", "DEF", "MID", "FWD"]:
        best = winners[pos_name]
        baseline_row = baseline_df[
            (baseline_df["position"] == pos_name) & (baseline_df["split"] == "unseen")
        ]
        baseline_mae = baseline_row.iloc[0]["mae"] if not baseline_row.empty else float("nan")
        improvement = baseline_mae - best["unseen_mae"]
        print(f"{pos_name:<10} {best['approach']:<35} {best['unseen_mae']:>11.2f} "
              f"{baseline_mae:>10.2f} {improvement:>+11.2f}")

    # ── Save winning models ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving winning models to {MODEL_DIR}/")
    for pos_name, best in winners.items():
        pos_code = POS_NAME_TO_CODE[pos_name]
        model_path = MODEL_DIR / f"position_model_{pos_code}.joblib"
        joblib.dump(best["_model"], model_path)
        print(f"  {pos_name} ({best['approach']}) → {model_path.name}")

    # ── Update selected_features.json to match winner's feature set ──
    # If the winner uses full features (Task4 single), store the full feature list
    # If the winner uses pruned features, store the pruned list
    final_features = {}
    for pos_name, best in winners.items():
        pos_code = POS_NAME_TO_CODE[pos_name]
        final_features[str(pos_code)] = best["_features"]

    features_path = TRAINING_DIR / "selected_features.json"
    with open(features_path, "w") as f:
        json.dump(final_features, f, indent=2)
    print(f"\nSelected features saved to {features_path}")

    # ── Save results CSV ──
    csv_cols = ["position", "approach", "algorithm", "n_features",
                "test_mae", "test_rmse", "test_r2",
                "unseen_mae", "unseen_rmse", "unseen_r2"]
    csv_df = results_df[csv_cols]
    csv_path = TRAINING_DIR / "final_comparison.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

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

    print("\nDone!")


if __name__ == "__main__":
    train_final_models()
