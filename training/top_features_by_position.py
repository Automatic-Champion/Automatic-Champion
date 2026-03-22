from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "historical_exports"
MODEL_DIR = BASE_DIR / "models"

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


def load_models() -> dict[int, RandomForestRegressor]:
    models: dict[int, RandomForestRegressor] = {}
    for position in POSITION_MAP:
        path = MODEL_DIR / f"position_model_{position}.joblib"
        if path.exists():
            models[position] = joblib.load(path)
    return models


def train_models(df_train: pd.DataFrame, feature_cols: list[str]) -> dict[int, RandomForestRegressor]:
    models: dict[int, RandomForestRegressor] = {}
    for position in POSITION_MAP:
        subset = df_train[df_train["element_type"] == position]
        if subset.empty:
            continue
        X_train = subset[feature_cols].fillna(0)
        y_train = subset["total_points"]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        models[position] = model
    return models


def main() -> None:
    df_train = pd.concat([load_season(season) for season in SEASONS_TRAIN], ignore_index=True)
    feature_cols = build_feature_list(df_train)
    if not feature_cols:
        raise ValueError("No feature columns found.")

    models = load_models()
    if len(models) != len(POSITION_MAP):
        models = train_models(df_train, feature_cols)

    for position, label in POSITION_MAP.items():
        model = models.get(position)
        if model is None:
            print(f"\n{label}: missing model")
            continue
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        top = importances.sort_values(ascending=False).head(5)
        print(f"\nTop 5 features for {label}:")
        for idx, (feat, score) in enumerate(top.items(), start=1):
            print(f"{idx}. {feat}: {score:.3f}")


if __name__ == "__main__":
    main()
