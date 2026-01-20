from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "historical_player_exports" / "output"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_PATH = BASE_DIR / "mae_by_position.png"

SEASONS_TRAIN = ["2019-20", "2020-21", "2021-22", "2022-23"]
SEASON_VAL = "2023-24"

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


def main() -> None:
    train_frames = [load_season(season) for season in SEASONS_TRAIN]
    df_train = pd.concat(train_frames, ignore_index=True)
    df_val = load_season(SEASON_VAL)

    if "element_type" not in df_train.columns or "element_type" not in df_val.columns:
        raise ValueError("Expected 'element_type' column in train/val data.")

    feature_cols = build_feature_list(df_train)
    if not feature_cols:
        raise ValueError("No feature columns found for training.")

    models: dict[int, RandomForestRegressor] = {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mae_by_position: dict[int, float] = {}

    for position in POSITION_MAP:
        train_pos = df_train[df_train["element_type"] == position]
        val_pos = df_val[df_val["element_type"] == position]

        if train_pos.empty or val_pos.empty:
            print(f"{POSITION_MAP[position]}: missing train/val data, skipping.")
            continue

        X_train_pos = train_pos[feature_cols].fillna(0)
        y_train_pos = train_pos["total_points"]
        X_val_pos = val_pos[feature_cols].fillna(0)
        y_val_pos = val_pos["total_points"]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_pos, y_train_pos)
        models[position] = model
        joblib.dump(model, MODEL_DIR / f"position_model_{position}.joblib")

        preds = model.predict(X_val_pos)
        mae = mean_absolute_error(y_val_pos, preds)
        mae_by_position[position] = mae
        print(f"{POSITION_MAP[position]} MAE: {mae:.2f}")

    positions = [POSITION_MAP[pos] for pos in mae_by_position.keys()]
    maes = [mae_by_position[pos] for pos in mae_by_position.keys()]

    plt.figure(figsize=(8, 5))
    plt.bar(positions, maes, color="#4C78A8")
    plt.title("Predictability Gap: Which Position is Hardest to Predict?")
    plt.xlabel("Position")
    plt.ylabel("Mean Absolute Error (Lower is Better)")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
