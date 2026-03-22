from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "historical_exports"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_PATH = BASE_DIR / "visuals" / "feature_importance_heatmap.png"

SEASONS_TRAIN = ["2019-20", "2020-21", "2021-22", "2022-23"]

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def load_season(season: str) -> pd.DataFrame:
    path = DATA_DIR / f"historical_players_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def build_feature_list(df: pd.DataFrame) -> list[str]:
    features = ["price_now"]
    for col in df.columns:
        if col.startswith("1_years_past_"):
            features.append(col)
    features = [col for col in features if col in df.columns]
    numeric_cols = df[features].select_dtypes(include="number").columns.tolist()
    return numeric_cols


def load_models() -> dict[int, object]:
    models: dict[int, object] = {}
    for position in POSITION_MAP:
        path = MODEL_DIR / f"position_model_{position}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path}")
        models[position] = joblib.load(path)
    return models


def main() -> None:
    df_train = pd.concat([load_season(season) for season in SEASONS_TRAIN], ignore_index=True)
    feature_cols = build_feature_list(df_train)
    if not feature_cols:
        raise ValueError("No feature columns found for feature importance.")

    models = load_models()

    top_features_by_pos: dict[str, pd.Series] = {}
    for position, label in POSITION_MAP.items():
        model = models[position]
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            raise ValueError(f"Model for {label} does not expose feature_importances_.")
        series = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
        top_features_by_pos[label] = series.head(10)

    all_features = sorted({feat for series in top_features_by_pos.values() for feat in series.index})
    table = pd.DataFrame(index=all_features, columns=POSITION_MAP.values(), dtype=float)
    for label, series in top_features_by_pos.items():
        for feat in all_features:
            table.at[feat, label] = series.get(feat, 0.0)

    print("\nTop Feature Importances by Position (Top 10 each):")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(table.fillna(0.0).round(4))

    plt.figure(figsize=(10, max(6, len(all_features) * 0.25)))
    plt.imshow(table.fillna(0.0).values, aspect="auto", cmap="viridis")
    plt.colorbar(label="Feature Importance")
    plt.yticks(range(len(all_features)), all_features)
    plt.xticks(range(len(POSITION_MAP)), list(POSITION_MAP.values()))
    plt.title("Top Feature Importances by Position")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
