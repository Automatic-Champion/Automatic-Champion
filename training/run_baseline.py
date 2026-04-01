"""Generate baseline metrics for the current position models.

Evaluates position_model_*.joblib on both the test (2022-23) and unseen (2023-24)
splits and saves results to training/baseline_metrics.csv.

Usage:
    python -m training.run_baseline
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so this works both as a module and as a script.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import joblib
import pandas as pd

from training.eval_framework import (
    BASE_DIR,
    POSITION_MAP,
    evaluate_all_positions,
    load_train_test_unseen,
)

MODEL_DIR = BASE_DIR / "models"
OUTPUT_CSV = BASE_DIR / "training" / "baseline_metrics.csv"

MODEL_FILENAMES = {
    1: "position_model_1.joblib",
    2: "position_model_2.joblib",
    3: "position_model_3.joblib",
    4: "position_model_4.joblib",
}


def build_feature_cols(df: pd.DataFrame) -> list[str]:
    """Build feature columns matching src/team_builder._build_feature_cols."""
    features = ["price_now"]
    for col in df.columns:
        if col.startswith("1_years_past_"):
            features.append(col)
    features = [col for col in features if col in df.columns]
    numeric_cols = df[features].select_dtypes(include="number").columns.tolist()
    return numeric_cols


def load_models() -> dict[int, object]:
    models = {}
    for pos_code, filename in MODEL_FILENAMES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path}")
        models[pos_code] = joblib.load(path)
        print(f"  Loaded {POSITION_MAP[pos_code]} model from {path.name}")
    return models


def main() -> None:
    print("Loading canonical data splits...")
    train_df, test_df, unseen_df = load_train_test_unseen()
    print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows, Unseen: {len(unseen_df)} rows")

    print("\nLoading models...")
    models = load_models()

    feature_cols = build_feature_cols(train_df)
    print(f"\nUsing {len(feature_cols)} features: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")

    print("\n=== Test Set (2022-23) ===")
    test_results = evaluate_all_positions(models, feature_cols, test_df)
    test_results["split"] = "test"

    print("\n=== Unseen Set (2023-24) ===")
    unseen_results = evaluate_all_positions(models, feature_cols, unseen_df)
    unseen_results["split"] = "unseen"

    baseline_df = pd.concat([test_results, unseen_results], ignore_index=True)
    baseline_df = baseline_df[["position", "split", "mae", "rmse", "r2", "median_ae", "n_samples"]]
    baseline_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nBaseline metrics saved to: {OUTPUT_CSV}")
    print("\n=== Full Baseline Table ===")
    print(baseline_df.to_string(index=False))


if __name__ == "__main__":
    main()
