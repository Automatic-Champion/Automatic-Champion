from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def build_feature_list(df: pd.DataFrame) -> list[str]:
    features = [col for col in df.columns if col.startswith("1_years_past")]
    features.append("price_now")
    if "transfers_in" in df.columns:
        features.append("transfers_in")
    blocked = {"total_points", "element_type"}
    features = [col for col in features if col in df.columns and col not in blocked]
    numeric_cols = df[features].select_dtypes(include="number").columns.tolist()
    return numeric_cols


def main(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    if "element_type" not in df_train.columns or "element_type" not in df_test.columns:
        raise ValueError("Expected 'element_type' in df_train and df_test.")

    feature_cols = build_feature_list(df_train)
    if not feature_cols:
        raise ValueError("No valid feature columns found.")

    mae_by_position: dict[int, float] = {}
    count_by_position: dict[int, int] = {}
    models: dict[int, XGBRegressor] = {}

    for position in POSITION_MAP:
        train_pos = df_train[df_train["element_type"] == position]
        test_pos = df_test[df_test["element_type"] == position]

        if train_pos.empty or test_pos.empty:
            print(f"{POSITION_MAP[position]}: missing train/test data, skipping.")
            continue

        X_train = train_pos[feature_cols].fillna(0)
        y_train = train_pos["total_points"]
        X_test = test_pos[feature_cols].fillna(0)
        y_test = test_pos.get("total_points")

        if position in (1, 2):
            max_depth = 4
            learning_rate = 0.1
            n_estimators = 200
        else:
            max_depth = 2
            learning_rate = 0.05
            n_estimators = 500

        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective="reg:squarederror",
            random_state=42,
        )
        model.fit(X_train, y_train, verbose=False)
        models[position] = model

        preds = model.predict(X_test)
        if y_test is None:
            print(f"{POSITION_MAP[position]}: missing total_points in df_test.")
            continue

        mae = mean_absolute_error(y_test, preds)
        mae_by_position[position] = mae
        count_by_position[position] = len(y_test)
        print(f"{POSITION_MAP[position]} MAE: {mae:.2f}")

    if mae_by_position:
        total_weight = sum(count_by_position.values())
        weighted_mae = sum(
            mae_by_position[pos] * count_by_position[pos] for pos in mae_by_position
        ) / total_weight
        print(f"Weighted Average MAE: {weighted_mae:.2f}")

    gk_model = models.get(1)
    if gk_model is not None:
        importances = pd.Series(
            gk_model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
        top10 = importances.head(10).sort_values(ascending=True)

        plt.figure(figsize=(8, 6))
        plt.barh(top10.index, top10.values, color="#4C78A8")
        plt.title("GK Feature Importance (Top 10)")
        plt.xlabel("Importance")
        plt.tight_layout()
        output_path = OUTPUT_DIR / "gk_feature_importance.png"
        plt.savefig(output_path, dpi=150)
        print(f"Saved GK feature importance plot to: {output_path}")


if __name__ == "__main__":
    raise SystemExit(
        "Import this module and call main(df_train, df_test) with your dataframes."
    )
