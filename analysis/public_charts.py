from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "historical_exports"
MODEL_DIR = BASE_DIR / "models"

STAR_PATH = BASE_DIR / "visuals" / "public_star_test.png"
SQUAD_PATH = BASE_DIR / "visuals" / "public_squad_score.png"

SEASONS_TRAIN = ["2019-20", "2020-21", "2021-22", "2022-23"]
SEASON_TEST = "2024-25"
HOUSEHOLD_NAMES = ["Haaland", "Salah", "Saka", "Palmer", "Foden", "Son", "Watkins", "Isak"]

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
PALETTE = {"Actual": "#0B1F3A", "Predicted": "#D4AF37"}


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


def add_predictions(df_results: pd.DataFrame) -> pd.DataFrame:
    if "predicted_points" in df_results.columns:
        return df_results
    df_train = pd.concat([load_season(season) for season in SEASONS_TRAIN], ignore_index=True)
    feature_cols = build_feature_list(df_train)
    if not feature_cols:
        raise ValueError("No feature columns found to build predictions.")
    models = load_models()
    if len(models) != len(POSITION_MAP):
        models = train_models(df_train, feature_cols)
    preds = pd.Series(index=df_results.index, dtype="float64")
    for position, model in models.items():
        mask = df_results["element_type"] == position
        subset = df_results.loc[mask, feature_cols].fillna(0)
        if subset.empty:
            continue
        preds.loc[mask] = model.predict(subset)
    df_results = df_results.copy()
    df_results["predicted_points"] = preds
    return df_results


def load_results() -> pd.DataFrame:
    df_results = load_season(SEASON_TEST)
    if "element_type" not in df_results.columns:
        raise ValueError("Expected 'element_type' column in the test data.")
    return add_predictions(df_results)


def build_name(df: pd.DataFrame) -> pd.Series:
    return (
        df[["first_name", "second_name"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.strip()
    )


def main() -> None:
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "sans-serif"

    df_results = load_results()
    df_results["name"] = build_name(df_results)

    mask = df_results["name"].str.contains("|".join(HOUSEHOLD_NAMES), case=False, na=False)
    df_stars = df_results.loc[mask, ["name", "total_points", "predicted_points"]].copy()
    df_stars = df_stars.rename(columns={"name": "Player"})
    df_long = df_stars.melt(
        id_vars="Player",
        value_vars=["total_points", "predicted_points"],
        var_name="Type",
        value_name="Points",
    )
    df_long["Type"] = df_long["Type"].map(
        {"total_points": "Actual", "predicted_points": "Predicted"}
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_long, x="Player", y="Points", hue="Type", palette=PALETTE)
    plt.title("Did the AI Predict the Stars?")
    plt.xlabel("Player")
    plt.ylabel("Points")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(STAR_PATH, dpi=200)

    df_sorted = df_results.sort_values("predicted_points", ascending=False)
    squad = []
    for position, count in [(1, 1), (2, 4), (3, 4), (4, 2)]:
        picks = df_sorted[df_sorted["element_type"] == position].head(count)
        squad.append(picks)
    df_squad = pd.concat(squad, ignore_index=True)
    df_squad["position_name"] = df_squad["element_type"].map(POSITION_MAP)
    df_squad["name"] = build_name(df_squad)

    ai_score = df_squad["total_points"].sum()
    benchmark_score = 2200

    plt.figure(figsize=(6, 5))
    bars = plt.bar(
        ["Benchmark (Good Human)", "AI Selected Team"],
        [benchmark_score, ai_score],
        color=["#9E9E9E", "#2CA02C"],
    )
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    plt.title("Fantasy Season Simulation: AI XI vs. Human Benchmark")
    plt.ylabel("Total Points")
    plt.tight_layout()
    plt.savefig(SQUAD_PATH, dpi=200)

    print("\nAI Squad (Predicted Top XI):")
    print(df_squad[["name", "position_name", "predicted_points", "total_points"]].to_string(index=False))
    print(f"\nSaved: {STAR_PATH}")
    print(f"Saved: {SQUAD_PATH}")


if __name__ == "__main__":
    main()
