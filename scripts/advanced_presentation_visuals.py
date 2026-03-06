from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "historical_player_exports" / "output"
MODEL_DIR = BASE_DIR / "models"

SCATTER_PATH = BASE_DIR / "advanced_presentation_scatter_comparison.png"
HEATMAP_PATH = BASE_DIR / "advanced_presentation_feature_heatmap.png"
BAR_PATH = BASE_DIR / "advanced_presentation_performance_bar.png"

SEASONS_TRAIN = ["2019-20", "2020-21", "2021-22", "2022-23"]
SEASON_TEST = "2024-25"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
PALETTE = {"GK": "#7f7f7f", "DEF": "#1f77b4", "MID": "#ff7f0e", "FWD": "#2ca02c"}


def load_season(season: str) -> pd.DataFrame:
    path = DATA_DIR / f"historical_players_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def add_per_90_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    minutes = pd.to_numeric(out.get("1_years_past_minutes"), errors="coerce").fillna(0)
    denom = minutes.replace(0, np.nan)

    total_points = pd.to_numeric(out.get("1_years_past_total_points"), errors="coerce").fillna(0)
    ict_index = pd.to_numeric(out.get("1_years_past_ict_index"), errors="coerce").fillna(0)

    out["1_years_past_total_points_per_90"] = ((total_points / denom) * 90).fillna(0.0)
    out["1_years_past_ict_index_per_90"] = ((ict_index / denom) * 90).fillna(0.0)
    return out


def build_feature_list(df: pd.DataFrame) -> list[str]:
    features = ["price_now"] + [col for col in df.columns if col.startswith("1_years_past_")]
    features = [col for col in features if col in df.columns]
    numeric_cols = (
        df[features]
        .apply(pd.to_numeric, errors="coerce")
        .select_dtypes(include="number")
        .columns.tolist()
    )
    return numeric_cols


def load_baseline_models() -> dict[int, object]:
    models: dict[int, object] = {}
    for position in POSITION_MAP:
        path = MODEL_DIR / f"position_model_{position}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing baseline model: {path}")
        models[position] = joblib.load(path)
    return models


def load_advanced_models() -> dict[int, object]:
    models: dict[int, object] = {}
    for code, label in POSITION_MAP.items():
        path = MODEL_DIR / f"advanced_model_{label}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing advanced model: {path}")
        models[code] = joblib.load(path)
    return models


def predict_by_position(
    df: pd.DataFrame,
    models: dict[int, object],
    feature_cols: list[str],
) -> pd.Series:
    preds = pd.Series(index=df.index, dtype="float64")
    for position, model in models.items():
        mask = df["element_type"] == position
        subset = df.loc[mask, feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        if subset.empty:
            continue
        preds.loc[mask] = model.predict(subset)
    return preds


def annotate_outliers(ax: plt.Axes, df: pd.DataFrame, error_col: str) -> None:
    if df.empty:
        return
    top_over = df.sort_values(error_col, ascending=False).head(3)
    top_under = df.sort_values(error_col, ascending=True).head(3)
    for _, row in pd.concat([top_over, top_under]).iterrows():
        ax.annotate(
            row["name"],
            (row["total_points"], row["predicted_points"]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            fontsize=8,
        )


def model_importance_series(model: object, feature_cols: list[str]) -> pd.Series:
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
        return pd.Series(vals, index=feature_cols)
    if hasattr(model, "coef_"):
        vals = np.asarray(model.coef_, dtype=float)
        if vals.ndim > 1:
            vals = vals.ravel()
        return pd.Series(np.abs(vals), index=feature_cols)
    return pd.Series(dtype=float)


def main() -> None:
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "sans-serif"

    df_train_base = pd.concat([load_season(season) for season in SEASONS_TRAIN], ignore_index=True)
    df_test_base = load_season(SEASON_TEST)

    if "element_type" not in df_test_base.columns:
        raise ValueError("Expected 'element_type' column in test data.")

    df_train_adv = add_per_90_features(df_train_base)
    df_test_adv = add_per_90_features(df_test_base)

    baseline_feature_cols = build_feature_list(df_train_base)
    advanced_feature_cols = build_feature_list(df_train_adv)
    if not baseline_feature_cols:
        raise ValueError("No baseline feature columns found.")
    if not advanced_feature_cols:
        raise ValueError("No advanced feature columns found.")

    baseline_models = load_baseline_models()
    advanced_models = load_advanced_models()

    baseline_results = df_test_base.copy()
    baseline_results["predicted_points"] = predict_by_position(
        baseline_results, baseline_models, baseline_feature_cols
    )
    baseline_results["error"] = baseline_results["total_points"] - baseline_results["predicted_points"]
    baseline_results["position_name"] = baseline_results["element_type"].map(POSITION_MAP)
    baseline_results["name"] = (
        baseline_results[["first_name", "second_name"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.strip()
    )

    advanced_results = df_test_adv.copy()
    advanced_results["predicted_points"] = predict_by_position(
        advanced_results, advanced_models, advanced_feature_cols
    )
    advanced_results["error"] = advanced_results["total_points"] - advanced_results["predicted_points"]
    advanced_results["position_name"] = advanced_results["element_type"].map(POSITION_MAP)
    advanced_results["name"] = (
        advanced_results[["first_name", "second_name"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.strip()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    ax_left = axes[0]
    sns.scatterplot(
        data=baseline_results,
        x="total_points",
        y="predicted_points",
        hue="position_name",
        palette=PALETTE,
        alpha=0.65,
        ax=ax_left,
    )
    ax_left.plot([0, 350], [0, 350], linestyle="--", color="black", linewidth=1)
    annotate_outliers(ax_left, baseline_results.dropna(subset=["predicted_points"]), "error")
    ax_left.set_title("Baseline Models: Predicted vs Actual (24/25)")
    ax_left.set_xlabel("Actual Points")
    ax_left.set_ylabel("Predicted Points")

    ax_right = axes[1]
    sns.scatterplot(
        data=advanced_results,
        x="total_points",
        y="predicted_points",
        hue="position_name",
        palette=PALETTE,
        alpha=0.65,
        ax=ax_right,
        legend=False,
    )
    ax_right.plot([0, 350], [0, 350], linestyle="--", color="black", linewidth=1)
    annotate_outliers(ax_right, advanced_results.dropna(subset=["predicted_points"]), "error")
    ax_right.set_title("Advanced Models: Predicted vs Actual (24/25)")
    ax_right.set_xlabel("Actual Points")
    ax_right.set_ylabel("")

    plt.suptitle("Model Accuracy Comparison (24/25 Season)", fontsize=14)
    plt.tight_layout()
    plt.savefig(SCATTER_PATH, dpi=200)

    importances = {}
    for code, label in POSITION_MAP.items():
        series = model_importance_series(advanced_models[code], advanced_feature_cols)
        if series.empty:
            continue
        importances[label] = series

    if importances:
        importance_df = pd.DataFrame(importances).fillna(0.0)
        importance_df["max_importance"] = importance_df.max(axis=1)
        top_features = (
            importance_df.sort_values("max_importance", ascending=False)
            .head(10)
            .drop(columns=["max_importance"])
        )

        plt.figure(figsize=(10, 6))
        sns.heatmap(top_features, cmap="viridis", annot=True, fmt=".2f")
        plt.title("Advanced Models: Key Drivers by Position")
        plt.tight_layout()
        plt.savefig(HEATMAP_PATH, dpi=200)

    baseline_model_mae = (
        baseline_results["total_points"] - baseline_results["predicted_points"]
    ).abs().mean()
    advanced_model_mae = (
        advanced_results["total_points"] - advanced_results["predicted_points"]
    ).abs().mean()
    naive = pd.to_numeric(
        baseline_results.get("1_years_past_total_points", 0), errors="coerce"
    ).fillna(0)
    naive_mae = (baseline_results["total_points"] - naive).abs().mean()

    perf_df = pd.DataFrame(
        {
            "Strategy": ["Naive Baseline", "Baseline Models", "Advanced Models"],
            "Error (MAE)": [naive_mae, baseline_model_mae, advanced_model_mae],
        }
    )

    plt.figure(figsize=(8, 5))
    colors = ["#9e9e9e", "#4c78a8", "#2ca02c"]
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
    plt.title("MAE Comparison: Naive vs Baseline vs Advanced")
    plt.ylabel("Mean Absolute Error")
    plt.tight_layout()
    plt.savefig(BAR_PATH, dpi=200)

    print(f"Saved: {SCATTER_PATH}")
    if importances:
        print(f"Saved: {HEATMAP_PATH}")
    else:
        print("Skipped heatmap: no feature importances available from advanced models.")
    print(f"Saved: {BAR_PATH}")
    print(f"Baseline model MAE: {baseline_model_mae:.2f}")
    print(f"Advanced model MAE: {advanced_model_mae:.2f}")


if __name__ == "__main__":
    main()
