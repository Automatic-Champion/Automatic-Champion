from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "historical_player_exports" / "output"
MODEL_DIR = BASE_DIR / "models"

SCATTER_PATH = BASE_DIR / "presentation_scatter_plot.png"
HEATMAP_PATH = BASE_DIR / "presentation_feature_heatmap.png"
BAR_PATH = BASE_DIR / "presentation_performance_bar.png"

SEASONS_TRAIN = ["2019-20", "2020-21", "2021-22", "2022-23"]
SEASON_TEST = "2024-25"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
PALETTE = {"GK": "grey", "DEF": "#1f77b4", "MID": "#ff7f0e", "FWD": "#2ca02c"}


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


def predict_by_position(
    df: pd.DataFrame, models: dict[int, RandomForestRegressor], feature_cols: list[str]
) -> pd.Series:
    preds = pd.Series(index=df.index, dtype="float64")
    for position, model in models.items():
        mask = df["element_type"] == position
        subset = df.loc[mask, feature_cols].fillna(0)
        if subset.empty:
            continue
        preds.loc[mask] = model.predict(subset)
    return preds


def annotate_outliers(ax: plt.Axes, df: pd.DataFrame) -> None:
    top_over = df.sort_values("error", ascending=False).head(3)
    top_under = df.sort_values("error", ascending=True).head(3)
    for _, row in pd.concat([top_over, top_under]).iterrows():
        ax.annotate(
            row["name"],
            (row["total_points"], row["predicted_points"]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
            fontsize=9,
        )


def main() -> None:
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "sans-serif"

    df_train = pd.concat([load_season(season) for season in SEASONS_TRAIN], ignore_index=True)
    df_test = load_season(SEASON_TEST)

    if "element_type" not in df_test.columns:
        raise ValueError("Expected 'element_type' column in test data.")

    feature_cols = build_feature_list(df_train)
    if not feature_cols:
        raise ValueError("No feature columns found for visualization.")

    models = load_models()
    if len(models) != len(POSITION_MAP):
        models = train_models(df_train, feature_cols)

    df_results = df_test.copy()
    df_results["predicted_points"] = predict_by_position(df_results, models, feature_cols)
    df_results["error"] = df_results["total_points"] - df_results["predicted_points"]
    df_results["position_name"] = df_results["element_type"].map(POSITION_MAP)
    df_results["name"] = (
        df_results[["first_name", "second_name"]].fillna("").agg(" ".join, axis=1).str.strip()
    )

    plt.figure(figsize=(9, 7))
    ax = sns.scatterplot(
        data=df_results,
        x="total_points",
        y="predicted_points",
        hue="position_name",
        palette=PALETTE,
        alpha=0.7,
    )
    ax.plot([0, 350], [0, 350], linestyle="--", color="grey")
    annotate_outliers(ax, df_results.dropna(subset=["predicted_points"]))
    ax.set_title("Model Accuracy: Predicted vs. Actual Points (24/25 Season)")
    ax.set_xlabel("Actual Points")
    ax.set_ylabel("Predicted Points")
    plt.tight_layout()
    plt.savefig(SCATTER_PATH, dpi=200)

    importances = {}
    for position, label in POSITION_MAP.items():
        model = models[position]
        series = pd.Series(model.feature_importances_, index=feature_cols)
        importances[label] = series
    importance_df = pd.DataFrame(importances)
    importance_df["max_importance"] = importance_df.max(axis=1)
    top_features = importance_df.sort_values("max_importance", ascending=False).head(10)
    top_features = top_features.drop(columns=["max_importance"])

    plt.figure(figsize=(10, 6))
    sns.heatmap(top_features, cmap="viridis", annot=True, fmt=".2f")
    plt.title("Strategic Differences: Key Drivers by Position")
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=200)

    model_mae = (df_results["total_points"] - df_results["predicted_points"]).abs().mean()
    baseline = pd.to_numeric(
        df_results.get("1_years_past_total_points", 0), errors="coerce"
    ).fillna(0)
    baseline_mae = (df_results["total_points"] - baseline).abs().mean()

    perf_df = pd.DataFrame(
        {
            "Strategy": ["Naive Baseline", "My AI Model"],
            "Error (MAE)": [baseline_mae, model_mae],
        }
    )

    plt.figure(figsize=(7, 5))
    colors = ["grey", "#2ca02c"]
    bars = plt.bar(perf_df["Strategy"], perf_df["Error (MAE)"], color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.title("Reducing Uncertainty: Model Error vs. Baseline")
    plt.ylabel("Mean Absolute Error")
    plt.tight_layout()
    plt.savefig(BAR_PATH, dpi=200)

    print(f"Saved: {SCATTER_PATH}")
    print(f"Saved: {HEATMAP_PATH}")
    print(f"Saved: {BAR_PATH}")


if __name__ == "__main__":
    main()
