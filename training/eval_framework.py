from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "historical_exports"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

TRAIN_SEASONS = ["2019-20", "2020-21", "2021-22"]
TEST_SEASON = "2022-23"
UNSEEN_SEASON = "2023-24"


def _load_season(season: str) -> pd.DataFrame:
    path = DATA_DIR / f"historical_players_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def load_train_test_unseen() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df, unseen_df) with fixed season splits.

    - train:  2019-20, 2020-21, 2021-22  (3 seasons)
    - test:   2022-23                      (1 season)
    - unseen: 2023-24                      (1 season — never used for tuning)

    Drops rows with NaN in 'total_points' or 'element_type'.
    Returns raw DataFrames — caller decides features.
    """
    train_frames = [_load_season(s) for s in TRAIN_SEASONS]
    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = _load_season(TEST_SEASON)
    unseen_df = _load_season(UNSEEN_SEASON)

    for df in (train_df, test_df, unseen_df):
        mask = df["total_points"].notna() & df["element_type"].notna()
        df.drop(df[~mask].index, inplace=True)

    return train_df, test_df, unseen_df


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, position_name: str) -> dict:
    """Evaluate a single model and return metrics dict.

    Returns dict with: mae, rmse, r2, median_ae, predictions, actuals, position.
    """
    predictions = model.predict(X_test)
    actuals = y_test.values

    return {
        "mae": float(mean_absolute_error(actuals, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(actuals, predictions))),
        "r2": float(r2_score(actuals, predictions)),
        "median_ae": float(median_absolute_error(actuals, predictions)),
        "predictions": predictions,
        "actuals": actuals,
        "position": position_name,
    }


def evaluate_all_positions(
    models: dict[int, object],
    feature_cols: list[str],
    test_df: pd.DataFrame,
    position_col: str = "element_type",
) -> pd.DataFrame:
    """Evaluate models for all positions on a test DataFrame.

    Takes a dict of {position_int: trained_model} and a test DataFrame.
    Returns a DataFrame with one row per position and columns:
    position, mae, rmse, r2, median_ae, n_samples.
    Prints a formatted table to stdout.
    """
    rows = []
    for pos_code, pos_name in POSITION_MAP.items():
        if pos_code not in models:
            print(f"  {pos_name}: no model provided, skipping.")
            continue

        pos_df = test_df[test_df[position_col] == pos_code]
        if pos_df.empty:
            print(f"  {pos_name}: no test data, skipping.")
            continue

        X = pos_df[feature_cols].fillna(0)
        y = pos_df["total_points"]

        result = evaluate_model(models[pos_code], X, y, pos_name)
        rows.append({
            "position": pos_name,
            "mae": result["mae"],
            "rmse": result["rmse"],
            "r2": result["r2"],
            "median_ae": result["median_ae"],
            "n_samples": len(y),
        })

    results_df = pd.DataFrame(rows)

    if not results_df.empty:
        print(f"\n{'Position':<10} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MedAE':>8} {'N':>6}")
        print("-" * 50)
        for _, row in results_df.iterrows():
            print(
                f"{row['position']:<10} {row['mae']:>8.2f} {row['rmse']:>8.2f} "
                f"{row['r2']:>8.3f} {row['median_ae']:>8.2f} {row['n_samples']:>6.0f}"
            )

    return results_df


def compare_to_baseline(
    results_df: pd.DataFrame,
    baseline_csv: str = "training/baseline_metrics.csv",
) -> pd.DataFrame:
    """Compare current results to the saved baseline.

    Loads the baseline metrics CSV and merges with current results.
    Returns a DataFrame showing per-position improvements/regressions.
    Prints a formatted comparison table with indicators.
    """
    baseline_path = Path(baseline_csv)
    if not baseline_path.is_absolute():
        baseline_path = BASE_DIR / baseline_csv
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {baseline_path}\n"
            "Run 'python training/run_baseline.py' first to generate it."
        )

    baseline_df = pd.read_csv(baseline_path)

    merged_rows = []
    for _, row in results_df.iterrows():
        pos = row["position"]
        for split in baseline_df["split"].unique():
            baseline_row = baseline_df[
                (baseline_df["position"] == pos) & (baseline_df["split"] == split)
            ]
            if baseline_row.empty:
                continue
            br = baseline_row.iloc[0]
            mae_diff = row["mae"] - br["mae"]
            rmse_diff = row["rmse"] - br["rmse"]
            merged_rows.append({
                "position": pos,
                "baseline_split": split,
                "baseline_mae": br["mae"],
                "new_mae": row["mae"],
                "mae_improvement": -mae_diff,
                "baseline_rmse": br["rmse"],
                "new_rmse": row["rmse"],
                "rmse_improvement": -rmse_diff,
            })

    comparison_df = pd.DataFrame(merged_rows)

    if not comparison_df.empty:
        print(f"\n{'Position':<10} {'Split':<8} {'Base MAE':>10} {'New MAE':>10} {'Δ MAE':>10} "
              f"{'Base RMSE':>10} {'New RMSE':>10} {'Δ RMSE':>10}")
        print("-" * 80)
        for _, row in comparison_df.iterrows():
            mae_indicator = "▼" if row["mae_improvement"] > 0 else "▲"
            rmse_indicator = "▼" if row["rmse_improvement"] > 0 else "▲"
            print(
                f"{row['position']:<10} {row['baseline_split']:<8} "
                f"{row['baseline_mae']:>10.2f} {row['new_mae']:>10.2f} "
                f"{row['mae_improvement']:>+9.2f}{mae_indicator} "
                f"{row['baseline_rmse']:>10.2f} {row['new_rmse']:>10.2f} "
                f"{row['rmse_improvement']:>+9.2f}{rmse_indicator}"
            )

    return comparison_df
