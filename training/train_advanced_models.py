from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - environment dependency
    raise ImportError(
        "xgboost is required for this script. Install with: pip install xgboost"
    ) from exc

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:  # pragma: no cover - environment dependency
    raise ImportError(
        "lightgbm is required for this script. Install with: pip install lightgbm"
    ) from exc


POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
DEFAULT_HOLDOUT = "2023-24"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare advanced regression models by FPL position, "
            "with per-90 features and weighted fitting."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/historical_exports",
        help="Directory containing historical_players_<season>.csv files.",
    )
    parser.add_argument(
        "--visuals-dir",
        type=str,
        default="visuals",
        help="Directory to save charts and output tables.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save winning advanced models.",
    )
    parser.add_argument(
        "--holdout-season",
        type=str,
        default=DEFAULT_HOLDOUT,
        help="Preferred holdout season (defaults to 2023-24).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of CV folds for quick hyperparameter search.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for model/search (use 1 in restricted environments).",
    )
    parser.add_argument(
        "--search-iter",
        type=int,
        default=5,
        help="Number of random search iterations for non-linear models.",
    )
    return parser.parse_args()


def season_sort_key(season: str) -> int:
    return int(season.split("-")[0])


def discover_seasons(data_dir: Path) -> list[str]:
    seasons: list[str] = []
    for path in data_dir.glob("historical_players_*.csv"):
        season = path.stem.replace("historical_players_", "")
        if len(season) == 7 and "-" in season:
            seasons.append(season)
    seasons = sorted(set(seasons), key=season_sort_key)
    if not seasons:
        raise FileNotFoundError(f"No historical season files found in: {data_dir}")
    return seasons


def pick_holdout(seasons: list[str], preferred: str) -> str:
    if preferred in seasons:
        return preferred
    return seasons[-1]


def load_season(data_dir: Path, season: str) -> pd.DataFrame:
    path = data_dir / f"historical_players_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    df = pd.read_csv(path)
    df["season"] = season
    return df


def add_per_90_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    minutes = pd.to_numeric(out.get("1_years_past_minutes"), errors="coerce").fillna(0)
    denom = minutes.replace(0, np.nan)

    total_points = pd.to_numeric(out.get("1_years_past_total_points"), errors="coerce").fillna(0)
    ict_index = pd.to_numeric(out.get("1_years_past_ict_index"), errors="coerce").fillna(0)

    out["1_years_past_total_points_per_90"] = ((total_points / denom) * 90).fillna(0.0)
    out["1_years_past_ict_index_per_90"] = ((ict_index / denom) * 90).fillna(0.0)
    return out


def build_feature_columns(df_train: pd.DataFrame) -> list[str]:
    candidate_cols = ["price_now"] + [
        col for col in df_train.columns if col.startswith("1_years_past_")
    ]
    candidate_cols = [col for col in candidate_cols if col in df_train.columns]

    if "price_now" not in candidate_cols:
        raise ValueError("Expected feature column 'price_now' was not found.")

    numeric_cols = (
        df_train[candidate_cols]
        .apply(pd.to_numeric, errors="coerce")
        .select_dtypes(include="number")
        .columns.tolist()
    )
    if not numeric_cols:
        raise ValueError("No numeric candidate features found.")
    return numeric_cols


def coerce_model_inputs(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["total_points"], errors="coerce")
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


def build_sample_weights(y_train: pd.Series) -> np.ndarray:
    # Emphasize high-scoring players to reduce premium-player underprediction.
    clipped = y_train.clip(lower=0)
    weights = clipped + 1.0
    return weights.to_numpy(dtype=float)


def model_registry(
    random_state: int, n_jobs: int
) -> dict[str, tuple[Pipeline, dict, str, bool]]:
    ridge = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            slice(0, None),
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("model", Ridge(random_state=random_state)),
        ]
    )

    rf = Pipeline(
        steps=[
            ("preprocess", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)),
        ]
    )

    xgb = Pipeline(
        steps=[
            ("preprocess", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=random_state,
                    n_jobs=n_jobs,
                    verbosity=0,
                ),
            ),
        ]
    )

    lgbm = Pipeline(
        steps=[
            ("preprocess", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(random_state=random_state, n_jobs=n_jobs, verbose=-1)),
        ]
    )

    return {
        "Ridge": (
            ridge,
            {"model__alpha": [0.1, 1.0, 10.0, 25.0]},
            "grid",
            False,
        ),
        "RandomForest": (
            rf,
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "random",
            True,
        ),
        "XGBoost": (
            xgb,
            {
                "model__n_estimators": [250, 500],
                "model__max_depth": [2, 4, 6],
                "model__learning_rate": [0.03, 0.08, 0.12],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
            "random",
            True,
        ),
        "LightGBM": (
            lgbm,
            {
                "model__n_estimators": [250, 500],
                "model__num_leaves": [15, 31, 63],
                "model__learning_rate": [0.03, 0.08, 0.12],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
            "random",
            True,
        ),
    }


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_feature_importance(
    fitted_pipeline: Pipeline,
    model_name: str,
    feature_cols: list[str],
) -> pd.Series:
    model = fitted_pipeline.named_steps["model"]

    if model_name == "Ridge":
        coefs = getattr(model, "coef_", None)
        if coefs is None:
            return pd.Series(dtype=float)
        return pd.Series(np.abs(coefs), index=feature_cols, dtype=float)

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.Series(dtype=float)
    return pd.Series(importances, index=feature_cols, dtype=float)


def fit_and_score_model(
    model_name: str,
    estimator: Pipeline,
    param_grid: dict,
    search_strategy: str,
    use_weights: bool,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int,
    random_state: int,
    n_jobs: int,
    search_iter: int,
    sample_weights: np.ndarray,
) -> dict:
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    if search_strategy == "grid":
        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
        )
    elif search_strategy == "random":
        search = RandomizedSearchCV(
            estimator=clone(estimator),
            param_distributions=param_grid,
            n_iter=search_iter,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported search strategy: {search_strategy}")

    fit_kwargs: dict = {}
    if use_weights:
        fit_kwargs["model__sample_weight"] = sample_weights

    search.fit(X_train, y_train, **fit_kwargs)

    best = search.best_estimator_
    preds = best.predict(X_test)

    mae_val = mean_absolute_error(y_test, preds)
    rmse_val = rmse(y_test, preds)

    return {
        "model": model_name,
        "best_estimator": best,
        "best_params": search.best_params_,
        "cv_mae": -float(search.best_score_),
        "test_mae": float(mae_val),
        "test_rmse": float(rmse_val),
        "predictions": preds,
        "used_sample_weight": use_weights,
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_model_showdown(metrics_df: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    positions = ["GK", "DEF", "MID", "FWD"]
    models = ["Ridge", "RandomForest", "XGBoost", "LightGBM"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.ravel()

    for idx, position in enumerate(positions):
        ax = axes[idx]
        data = (
            metrics_df[metrics_df["position"] == position]
            .set_index("model")
            .reindex(models)
            .reset_index()
        )
        sns.barplot(data=data, x="model", y="test_mae", ax=ax, palette="Set2")
        ax.set_title(f"{position} - Test MAE")
        ax.set_xlabel("")
        ax.set_ylabel("MAE")
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Advanced Model Showdown: MAE by Position", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_winner_scatter(
    winner_predictions: pd.DataFrame,
    out_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    positions = ["GK", "DEF", "MID", "FWD"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)
    axes = axes.ravel()

    for idx, position in enumerate(positions):
        ax = axes[idx]
        data = winner_predictions[winner_predictions["position"] == position]
        if data.empty:
            ax.set_title(f"{position} - no test rows")
            ax.axis("off")
            continue

        sns.scatterplot(data=data, x="actual_points", y="predicted_points", ax=ax, alpha=0.7)
        min_axis = min(data["actual_points"].min(), data["predicted_points"].min())
        max_axis = max(data["actual_points"].max(), data["predicted_points"].max())
        ax.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", color="black", linewidth=1)
        winner_model = data["winning_model"].iloc[0]
        ax.set_title(f"{position} winner: {winner_model}")
        ax.set_xlabel("Actual Total Points")
        ax.set_ylabel("Predicted Total Points")

    fig.suptitle("Advanced Accuracy Scatter: Winning Model per Position", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    top_features_df: pd.DataFrame,
    out_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    positions = ["GK", "DEF", "MID", "FWD"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, position in enumerate(positions):
        ax = axes[idx]
        subset = top_features_df[top_features_df["position"] == position].copy()
        if subset.empty:
            ax.set_title(f"{position} - no importances")
            ax.axis("off")
            continue

        subset = subset.sort_values("importance", ascending=True)
        sns.barplot(data=subset, x="importance", y="feature", ax=ax, palette="viridis")
        winner_model = subset["winning_model"].iloc[0]
        ax.set_title(f"{position} winner: {winner_model}")
        ax.set_xlabel("Importance")
        ax.set_ylabel("")

    fig.suptitle("Advanced Top 10 Features of Winning Model per Position", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    visuals_dir = Path(args.visuals_dir)
    models_dir = Path(args.models_dir)

    ensure_dir(visuals_dir)
    ensure_dir(models_dir)

    seasons = discover_seasons(data_dir)
    holdout_season = pick_holdout(seasons, args.holdout_season)
    train_seasons = [season for season in seasons if season < holdout_season]

    if not train_seasons:
        raise ValueError(f"No seasons available before holdout season {holdout_season}.")

    print(f"Detected seasons: {seasons}")
    print(f"Using holdout test season: {holdout_season}")
    print(f"Training seasons: {train_seasons}")

    train_df = pd.concat([load_season(data_dir, season) for season in train_seasons], ignore_index=True)
    test_df = load_season(data_dir, holdout_season)

    train_df = add_per_90_features(train_df)
    test_df = add_per_90_features(test_df)

    required = {"total_points", "element_type", "price_now"}
    missing = sorted(required - set(train_df.columns))
    if missing:
        raise ValueError(f"Training data missing required columns: {missing}")

    feature_cols = build_feature_columns(train_df)
    print(f"Using {len(feature_cols)} features (including engineered per-90 features).")

    all_metrics: list[dict] = []
    winner_rows: list[dict] = []
    top_features_rows: list[dict] = []

    registry = model_registry(args.random_state, args.n_jobs)

    for pos_code, pos_name in POSITION_MAP.items():
        train_pos = train_df[train_df["element_type"] == pos_code].copy()
        test_pos = test_df[test_df["element_type"] == pos_code].copy()

        if train_pos.empty or test_pos.empty:
            print(f"Skipping {pos_name}: missing train/test rows.")
            continue

        X_train, y_train = coerce_model_inputs(train_pos, feature_cols)
        X_test, y_test = coerce_model_inputs(test_pos, feature_cols)

        if X_train.empty or X_test.empty:
            print(f"Skipping {pos_name}: no usable numeric rows after coercion.")
            continue

        sample_weights = build_sample_weights(y_train)

        print(f"\n=== Position: {pos_name} (train={len(X_train)}, test={len(X_test)}) ===")

        position_results: list[dict] = []
        for model_name, (estimator, param_grid, search_strategy, use_weights) in registry.items():
            result = fit_and_score_model(
                model_name=model_name,
                estimator=estimator,
                param_grid=param_grid,
                search_strategy=search_strategy,
                use_weights=use_weights,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv_folds=args.cv_folds,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                search_iter=args.search_iter,
                sample_weights=sample_weights,
            )
            position_results.append(result)

            all_metrics.append(
                {
                    "position": pos_name,
                    "model": model_name,
                    "cv_mae": result["cv_mae"],
                    "test_mae": result["test_mae"],
                    "test_rmse": result["test_rmse"],
                    "best_params": str(result["best_params"]),
                    "used_sample_weight": result["used_sample_weight"],
                    "train_rows": int(len(X_train)),
                    "test_rows": int(len(X_test)),
                }
            )
            print(
                f"{model_name:<12} | test MAE={result['test_mae']:.2f} | "
                f"test RMSE={result['test_rmse']:.2f}"
            )

        winner = min(position_results, key=lambda row: row["test_mae"])
        winner_model = winner["model"]
        winner_preds = winner["predictions"]

        winner_rows.extend(
            {
                "position": pos_name,
                "winning_model": winner_model,
                "actual_points": float(actual),
                "predicted_points": float(pred),
            }
            for actual, pred in zip(y_test.to_numpy(), winner_preds)
        )

        importances = get_feature_importance(
            fitted_pipeline=winner["best_estimator"],
            model_name=winner_model,
            feature_cols=feature_cols,
        )
        if not importances.empty:
            top10 = importances.sort_values(ascending=False).head(10)
            for feature, value in top10.items():
                top_features_rows.append(
                    {
                        "position": pos_name,
                        "winning_model": winner_model,
                        "feature": feature,
                        "importance": float(value),
                    }
                )

        model_out = models_dir / f"advanced_model_{pos_name}.joblib"
        joblib.dump(winner["best_estimator"], model_out)

        print(f"Winner for {pos_name}: {winner_model} (MAE={winner['test_mae']:.2f})")
        print(f"Saved model: {model_out}")

    if not all_metrics:
        raise ValueError("No results were produced. Check data integrity by position.")

    metrics_df = pd.DataFrame(all_metrics)
    winners_df = pd.DataFrame(winner_rows)
    top_features_df = pd.DataFrame(top_features_rows)

    metrics_path = visuals_dir / "advanced_model_metrics_by_position.csv"
    winners_path = visuals_dir / "advanced_winner_predictions.csv"
    features_path = visuals_dir / "advanced_winner_top_features.csv"

    metrics_df.to_csv(metrics_path, index=False)
    winners_df.to_csv(winners_path, index=False)
    if not top_features_df.empty:
        top_features_df.to_csv(features_path, index=False)

    showdown_png = visuals_dir / "advanced_model_showdown_mae.png"
    scatter_png = visuals_dir / "advanced_winning_model_accuracy_scatter.png"
    importance_png = visuals_dir / "advanced_winning_model_top_features.png"

    plot_model_showdown(metrics_df, showdown_png)
    plot_winner_scatter(winners_df, scatter_png)
    if not top_features_df.empty:
        plot_feature_importance(top_features_df, importance_png)

    leaderboard = (
        metrics_df.sort_values(["position", "test_mae", "test_rmse"])
        [["position", "model", "test_mae", "test_rmse", "used_sample_weight"]]
        .reset_index(drop=True)
    )
    print("\n=== Advanced Test-Set Leaderboard (lower is better) ===")
    print(leaderboard.to_string(index=False))

    print("\nSaved outputs:")
    print(f"- {metrics_path}")
    print(f"- {winners_path}")
    if not top_features_df.empty:
        print(f"- {features_path}")
    print(f"- {showdown_png}")
    print(f"- {scatter_png}")
    if not top_features_df.empty:
        print(f"- {importance_png}")


if __name__ == "__main__":
    main()
