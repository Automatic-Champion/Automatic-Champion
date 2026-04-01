"""Train position models with expanded features (multi-year lags + momentum).

Usage: python training/train_expanded_features.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from training.eval_framework import (
    POSITION_MAP,
    compare_to_baseline,
    evaluate_all_positions,
    load_train_test_unseen,
)
from src.team_builder import _add_momentum_features

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


def build_expanded_feature_list(df: pd.DataFrame) -> list[str]:
    """Return feature columns: price_now + 1/2/3_years_past_* + momentum_*."""
    features = ["price_now"]
    for col in df.columns:
        if any(col.startswith(f"{y}_years_past_") for y in (1, 2, 3)):
            features.append(col)
        elif col.startswith("momentum_"):
            features.append(col)
    features = [col for col in features if col in df.columns]
    numeric_cols = df[features].select_dtypes(include="number").columns.tolist()
    return numeric_cols


def main() -> None:
    print("Loading data splits...")
    train_df, test_df, unseen_df = load_train_test_unseen()

    # Add momentum features to all splits
    for label, df in [("train", train_df), ("test", test_df), ("unseen", unseen_df)]:
        _add_momentum_features(df)
        print(f"  {label}: {len(df)} rows")

    feature_cols = build_expanded_feature_list(train_df)
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models: dict[int, RandomForestRegressor] = {}

    print("\n--- Training ---")
    for pos_code, pos_name in POSITION_MAP.items():
        train_pos = train_df[train_df["element_type"] == pos_code]
        if train_pos.empty:
            print(f"  {pos_name}: no training data, skipping.")
            continue

        X_train = train_pos[feature_cols].fillna(0)
        y_train = train_pos["total_points"]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        models[pos_code] = model

        model_path = MODEL_DIR / f"position_model_{pos_code}.joblib"
        joblib.dump(model, model_path)
        print(f"  {pos_name}: trained on {len(train_pos)} samples → {model_path.name}")

    # Evaluate on test split
    print("\n=== Test Split (2022-23) ===")
    test_results = evaluate_all_positions(models, feature_cols, test_df)

    # Evaluate on unseen split
    print("\n=== Unseen Split (2023-24) ===")
    unseen_results = evaluate_all_positions(models, feature_cols, unseen_df)

    # Compare to baseline
    print("\n=== Comparison to Baseline (test split) ===")
    compare_to_baseline(test_results)

    print("\n=== Comparison to Baseline (unseen split) ===")
    compare_to_baseline(unseen_results)

    print("\nDone. Models saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
