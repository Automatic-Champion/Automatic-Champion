from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "historical_player_exports" / "output"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_PATH = BASE_DIR / "rank_correlation_top50.png"

SEASONS_TRAIN = ["2019-20", "2020-21", "2021-22", "2022-23"]
SEASON_TEST = "2024-25"

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
    df_test = load_season(SEASON_TEST)

    if "element_type" not in df_test.columns:
        raise ValueError("Expected 'element_type' column in test data.")

    feature_cols = build_feature_list(df_train)
    if not feature_cols:
        raise ValueError("No feature columns found for testing.")

    models = load_models()

    df_results = df_test.copy()
    df_results["predicted_points"] = pd.NA

    for position, model in models.items():
        mask = df_results["element_type"] == position
        subset = df_results.loc[mask, feature_cols].fillna(0)
        if subset.empty:
            continue
        preds = model.predict(subset)
        df_results.loc[mask, "predicted_points"] = preds

    df_results["predicted_points"] = pd.to_numeric(df_results["predicted_points"], errors="coerce")
    df_results["naive_predicted_points"] = pd.to_numeric(
        df_results.get("1_years_past_total_points", 0), errors="coerce"
    ).fillna(0)

    valid_rows = df_results.dropna(subset=["predicted_points"]).copy()
    if valid_rows.empty:
        raise ValueError("No predictions available to evaluate.")

    model_mae = (valid_rows["total_points"] - valid_rows["predicted_points"]).abs().mean()
    baseline_mae = (valid_rows["total_points"] - valid_rows["naive_predicted_points"]).abs().mean()
    improvement = baseline_mae - model_mae

    print(f"Random Forest MAE: {model_mae:.2f}")
    print(f"Naive Baseline MAE: {baseline_mae:.2f}")
    print(f"Improvement over Baseline: {improvement:.2f} points")

    spearman = valid_rows[["predicted_points", "total_points"]].corr(
        method="spearman"
    ).iloc[0, 1]
    print(f"Spearman Rank Correlation: {spearman:.3f}")

    name_parts = (
        valid_rows[["first_name", "second_name"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.strip()
    )
    valid_rows["name"] = name_parts

    top_actual = (
        valid_rows.sort_values("total_points", ascending=False).head(20)["name"].tolist()
    )
    top_pred = (
        valid_rows.sort_values("predicted_points", ascending=False)
        .head(20)["name"]
        .tolist()
    )
    overlap = len(set(top_actual) & set(top_pred))
    print(f"Top 20 Overlap: {overlap}/20 players")

    top50 = valid_rows.sort_values("total_points", ascending=False).head(50).copy()
    top50["actual_rank"] = top50["total_points"].rank(ascending=False, method="min")
    top50["pred_rank"] = top50["predicted_points"].rank(ascending=False, method="min")

    plt.figure(figsize=(7, 6))
    plt.scatter(top50["pred_rank"], top50["actual_rank"], alpha=0.7, color="#4C78A8")
    max_rank = max(top50["pred_rank"].max(), top50["actual_rank"].max())
    plt.plot([1, max_rank], [1, max_rank], linestyle="--", color="gray", label="Perfect")
    plt.xlabel("Predicted Rank (Top 50)")
    plt.ylabel("Actual Rank (Top 50)")
    plt.title("Top 50 Rank Correlation: Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
