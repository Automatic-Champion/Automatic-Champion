from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

try:
    import joblib
except ImportError:  # pragma: no cover - optional dependency
    joblib = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "historical_exports"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_PATH = BASE_DIR / "visuals" / "final_24_25_performance.png"
OUTPUT_NO_ZERO_PATH = BASE_DIR / "visuals" / "final_24_25_performance_no_zero.png"

SEASONS_TRAIN = ["2019-20", "2020-21", "2021-22", "2022-23"]
SEASON_TEST = "2024-25"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
COLOR_MAP = {1: "red", 2: "blue", 3: "green", 4: "orange"}


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


def load_models() -> dict[int, RandomForestRegressor]:
    models: dict[int, RandomForestRegressor] = {}
    if joblib is None or not MODEL_DIR.exists():
        return models
    for position in POSITION_MAP:
        path = MODEL_DIR / f"position_model_{position}.joblib"
        if path.exists():
            models[position] = joblib.load(path)
    return models


def plot_scatter(df: pd.DataFrame, path: Path, title: str) -> None:
    plt.figure(figsize=(8, 6))
    for pos, color in COLOR_MAP.items():
        subset = df[df["element_type"] == pos]
        if subset.empty:
            continue
        plt.scatter(
            subset["total_points"],
            subset["predicted_points"],
            color=color,
            alpha=0.6,
            label=POSITION_MAP[pos],
        )

    min_val = pd.concat([df["total_points"], df["predicted_points"]]).min()
    max_val = pd.concat([df["total_points"], df["predicted_points"]]).max()
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", label="Perfect")

    plt.xlabel("Actual Points")
    plt.ylabel("Predicted Points")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)


def print_summary(df: pd.DataFrame, label: str, display_cols: list[str]) -> None:
    top_over = df.sort_values("error", ascending=False).head(10)
    top_under = df.sort_values("error", ascending=True).head(10)

    print(f"\nTop 10 Overperformers (Sleepers) - {label}:")
    print(top_over[display_cols].to_string(index=False))

    print(f"\nTop 10 Underperformers (Busts) - {label}:")
    print(top_under[display_cols].to_string(index=False))

    valid_rows = df.dropna(subset=["predicted_points"])
    r2 = r2_score(valid_rows["total_points"], valid_rows["predicted_points"])
    mse = mean_squared_error(valid_rows["total_points"], valid_rows["predicted_points"])
    rmse = mse ** 0.5
    print(f"\nOverall R2 ({label}): {r2:.3f}")
    print(f"Overall RMSE ({label}): {rmse:.2f}")


def main() -> None:
    df_train = pd.concat([load_season(season) for season in SEASONS_TRAIN], ignore_index=True)
    df_test = load_season(SEASON_TEST)

    if "element_type" not in df_train.columns or "element_type" not in df_test.columns:
        raise ValueError("Expected 'element_type' column in train/test data.")

    feature_cols = build_feature_list(df_train)
    if not feature_cols:
        raise ValueError("No feature columns found for testing.")

    models = load_models()
    if len(models) != len(POSITION_MAP):
        models = train_models(df_train, feature_cols)

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
    df_results["error"] = df_results["total_points"] - df_results["predicted_points"]

    name_parts = (
        df_results[["first_name", "second_name"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.strip()
    )
    df_results["name"] = name_parts
    df_results["position_name"] = df_results["element_type"].map(POSITION_MAP)

    if "team" not in df_results.columns:
        df_results["team"] = "N/A"

    display_cols = [
        "name",
        "team",
        "position_name",
        "predicted_points",
        "total_points",
        "error",
    ]

    print_summary(df_results, "All Players", display_cols)
    plot_scatter(
        df_results,
        OUTPUT_PATH,
        "2024-25 Final Performance: Actual vs Predicted (All Players)",
    )
    print(f"\nSaved plot to: {OUTPUT_PATH}")

    df_no_zero = df_results[df_results["total_points"] != 0].copy()
    if df_no_zero.empty:
        print("\nNo rows with non-zero total_points for filtered analysis.")
        return
    print_summary(df_no_zero, "Non-Zero Total Points", display_cols)
    plot_scatter(
        df_no_zero,
        OUTPUT_NO_ZERO_PATH,
        "2024-25 Final Performance: Actual vs Predicted (Non-Zero Only)",
    )
    print(f"\nSaved plot to: {OUTPUT_NO_ZERO_PATH}")


if __name__ == "__main__":
    main()
