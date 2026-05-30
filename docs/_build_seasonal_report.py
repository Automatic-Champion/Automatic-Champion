"""Generate charts and metrics for the seasonal model report.

Reads the saved models in models/position_model_*.joblib, evaluates them on the
held-out 2024-25 season (never used in training, hyperparameter tuning, or the
training framework's "unseen" split), and writes PNG charts to
docs/seasonal_model_images/.

This is read-only with respect to src/ and training/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.team_builder import _add_momentum_features  # noqa: E402

DATA_DIR = BASE_DIR / "data" / "historical_exports"
MODELS_DIR = BASE_DIR / "models"
FEATURES_PATH = BASE_DIR / "training" / "selected_features.json"
IMG_DIR = BASE_DIR / "docs" / "seasonal_model_images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
CODE_FROM_NAME = {v: k for k, v in POSITION_MAP.items()}
POS_COLOR = {"GK": "#E15759", "DEF": "#4E79A7", "MID": "#59A14F", "FWD": "#F28E2B"}

SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]
EVAL_SEASON = "2024-25"

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def load_season(season: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"historical_players_{season}.csv")
    df = df[df["total_points"].notna() & df["element_type"].notna()].copy()
    df["season"] = season
    return df


# ── Load data ────────────────────────────────────────────────────────────────

all_seasons = {s: load_season(s) for s in SEASONS}
all_df = pd.concat(all_seasons.values(), ignore_index=True)
all_df["position"] = all_df["element_type"].map(POSITION_MAP)

eval_df = all_seasons[EVAL_SEASON].copy()
eval_df = _add_momentum_features(eval_df)
eval_df["position"] = eval_df["element_type"].map(POSITION_MAP)

with open(FEATURES_PATH) as f:
    selected_features = json.load(f)

models = {}
for code, name in POSITION_MAP.items():
    models[name] = joblib.load(MODELS_DIR / f"position_model_{code}.joblib")

ALGO_NAME = {
    "GK": type(models["GK"]).__name__,
    "DEF": type(models["DEF"]).__name__,
    "MID": type(models["MID"]).__name__,
    "FWD": type(models["FWD"]).__name__,
}

# ── Compute predictions & metrics per position ────────────────────────────────

metric_rows = []
prediction_frames = {}

for pos_name, model in models.items():
    pos_code = CODE_FROM_NAME[pos_name]
    pos_df = eval_df[eval_df["element_type"] == pos_code].copy()
    feat_list = [c for c in selected_features[str(pos_code)] if c in pos_df.columns]
    X = pos_df[feat_list].fillna(0)
    y_true = pos_df["total_points"].astype(float).values
    y_pred = model.predict(X)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    metric_rows.append({
        "position": pos_name,
        "algorithm": ALGO_NAME[pos_name],
        "n_features": len(feat_list),
        "n_samples": len(pos_df),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    })

    prediction_frames[pos_name] = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "residual": y_pred - y_true,
        "position": pos_name,
    })

metrics_df = pd.DataFrame(metric_rows)
print("=" * 60)
print(f"Held-out evaluation on {EVAL_SEASON} season")
print("=" * 60)
print(metrics_df.to_string(index=False))

# ── Baseline comparison ──────────────────────────────────────────────────────

baseline_rows = []
for pos_name in POSITION_MAP.values():
    pos_code = CODE_FROM_NAME[pos_name]
    pos_df = eval_df[eval_df["element_type"] == pos_code].copy()
    y_true = pos_df["total_points"].astype(float).values

    # Baseline 1: last season's total_points unchanged
    last_season_pts = pd.to_numeric(
        pos_df["1_years_past_total_points"], errors="coerce"
    ).fillna(0).values
    b1_mae = float(mean_absolute_error(y_true, last_season_pts))

    # Baseline 2: predict the position mean from prior seasons
    train_mask = (all_df["season"] != EVAL_SEASON) & (
        all_df["element_type"] == pos_code
    )
    pos_mean = float(all_df.loc[train_mask, "total_points"].mean())
    b2_mae = float(mean_absolute_error(y_true, np.full_like(y_true, pos_mean)))

    baseline_rows.append({
        "position": pos_name,
        "model_mae": metrics_df.loc[
            metrics_df["position"] == pos_name, "mae"
        ].values[0],
        "baseline_last_season_mae": b1_mae,
        "baseline_position_mean_mae": b2_mae,
    })
baseline_df = pd.DataFrame(baseline_rows)
print("\nBaseline comparison (MAE, lower is better):")
print(baseline_df.to_string(index=False))

# ── Chart 1: players per season ───────────────────────────────────────────────

counts = all_df.groupby("season").size().reindex(SEASONS)
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(counts.index, counts.values, color="#4E79A7")
ax.set_title("FPL players per season in the training dataset")
ax.set_xlabel("Season")
ax.set_ylabel("Number of players")
for bar, value in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 8, f"{value}", ha="center", fontsize=10)
fig.tight_layout()
fig.savefig(IMG_DIR / "players_per_season.png")
plt.close(fig)

# ── Chart 2: distribution of total_points across all seasons ──────────────────

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(all_df["total_points"].dropna(), bins=40, color="#4E79A7", edgecolor="white")
median_pts = float(all_df["total_points"].median())
ax.axvline(median_pts, color="black", linestyle="--", linewidth=1)
ax.text(median_pts + 4, ax.get_ylim()[1] * 0.92, f"median = {median_pts:.0f}", fontsize=10)
ax.set_title("Distribution of season total points (all players, all seasons)")
ax.set_xlabel("Total points in a season")
ax.set_ylabel("Number of players")
fig.tight_layout()
fig.savefig(IMG_DIR / "points_distribution.png")
plt.close(fig)

# ── Chart 3: points by position (boxplot) ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4.5))
positions_order = ["GK", "DEF", "MID", "FWD"]
data_by_pos = [
    all_df.loc[all_df["position"] == p, "total_points"].dropna().values
    for p in positions_order
]
bp = ax.boxplot(
    data_by_pos,
    labels=positions_order,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
)
for patch, p in zip(bp["boxes"], positions_order):
    patch.set_facecolor(POS_COLOR[p])
    patch.set_alpha(0.75)
ax.set_title("Season points by position (all seasons combined)")
ax.set_xlabel("Position")
ax.set_ylabel("Total points in a season")
fig.tight_layout()
fig.savefig(IMG_DIR / "points_by_position.png")
plt.close(fig)

# ── Chart 4: actual vs predicted per position ─────────────────────────────────

for pos_name, pf in prediction_frames.items():
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(pf["y_true"], pf["y_pred"], alpha=0.55, color=POS_COLOR[pos_name], s=35)
    lo = float(min(pf["y_true"].min(), pf["y_pred"].min())) - 5
    hi = float(max(pf["y_true"].max(), pf["y_pred"].max())) + 5
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, label="perfect prediction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual season points")
    ax.set_ylabel("Predicted season points")
    mae = metrics_df.loc[metrics_df["position"] == pos_name, "mae"].values[0]
    r2 = metrics_df.loc[metrics_df["position"] == pos_name, "r2"].values[0]
    algo = ALGO_NAME[pos_name]
    ax.set_title(f"{pos_name} – actual vs predicted ({algo})\nMAE={mae:.2f}, R²={r2:.2f}, n={len(pf)}")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / f"actual_vs_predicted_{pos_name}.png")
    plt.close(fig)

# ── Chart 5: combined residual histogram ─────────────────────────────────────

all_residuals = pd.concat(prediction_frames.values(), ignore_index=True)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(all_residuals["residual"], bins=40, color="#4E79A7", edgecolor="white")
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_title("Prediction errors across all positions (held-out 2024-25)")
ax.set_xlabel("Predicted minus actual points  (positive = over-predicted)")
ax.set_ylabel("Number of players")
fig.tight_layout()
fig.savefig(IMG_DIR / "residuals_distribution.png")
plt.close(fig)

# ── Chart 6: MAE by position ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4.5))
order = ["GK", "DEF", "MID", "FWD"]
maes = [metrics_df.loc[metrics_df["position"] == p, "mae"].values[0] for p in order]
bars = ax.bar(order, maes, color=[POS_COLOR[p] for p in order])
for bar, value in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.1f}", ha="center", fontsize=11)
ax.set_title("Average prediction error per position (held-out 2024-25)")
ax.set_xlabel("Position")
ax.set_ylabel("Mean Absolute Error in points (lower is better)")
fig.tight_layout()
fig.savefig(IMG_DIR / "mae_by_position.png")
plt.close(fig)

# ── Chart 7: feature importance per position ─────────────────────────────────

def get_importances(model, feature_names):
    """Return a (feature, importance) DataFrame sorted descending."""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        imp = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        return None
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False)


def prettify(name: str) -> str:
    name = name.replace("_years_past_", "y ago ")
    name = name.replace("1y ago", "last yr")
    name = name.replace("2y ago", "2 yrs ago")
    name = name.replace("3y ago", "3 yrs ago")
    name = name.replace("gw_", "")
    name = name.replace("_", " ")
    return name


for pos_name in POSITION_MAP.values():
    pos_code = CODE_FROM_NAME[pos_name]
    feat_list = [c for c in selected_features[str(pos_code)] if c in eval_df.columns]
    imp_df = get_importances(models[pos_name], feat_list)
    if imp_df is None or imp_df["importance"].sum() == 0:
        continue
    top = imp_df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [prettify(f) for f in top["feature"]],
        top["importance"].values,
        color=POS_COLOR[pos_name],
    )
    ax.set_title(f"{pos_name} – top 10 features ({ALGO_NAME[pos_name]})")
    ax.set_xlabel(
        "Importance"
        + (" (|coefficient|)" if not hasattr(models[pos_name], "feature_importances_") else "")
    )
    fig.tight_layout()
    fig.savefig(IMG_DIR / f"feature_importance_{pos_name}.png")
    plt.close(fig)

# ── Chart 8: MAE vs baselines ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8.5, 4.8))
x = np.arange(len(order))
width = 0.27
model_maes = [baseline_df.loc[baseline_df["position"] == p, "model_mae"].values[0] for p in order]
last_maes = [baseline_df.loc[baseline_df["position"] == p, "baseline_last_season_mae"].values[0] for p in order]
mean_maes = [baseline_df.loc[baseline_df["position"] == p, "baseline_position_mean_mae"].values[0] for p in order]
ax.bar(x - width, mean_maes, width, label="Baseline: position average", color="#BAB0AC")
ax.bar(x, last_maes, width, label="Baseline: last season's points", color="#9C9CB5")
ax.bar(x + width, model_maes, width, label="Our model", color="#4E79A7")
ax.set_xticks(x)
ax.set_xticklabels(order)
ax.set_ylabel("Mean Absolute Error in points (lower is better)")
ax.set_title("Model vs naive baselines (held-out 2024-25)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(IMG_DIR / "mae_vs_baselines.png")
plt.close(fig)

# ── Persist metrics for the report writer ─────────────────────────────────────

out_path = IMG_DIR.parent / "_seasonal_metrics.json"
out = {
    "eval_season": EVAL_SEASON,
    "metrics": metrics_df.to_dict(orient="records"),
    "baselines": baseline_df.to_dict(orient="records"),
    "algorithm": ALGO_NAME,
    "season_counts": {s: int(len(all_seasons[s])) for s in SEASONS},
    "target_stats": {
        "median": float(all_df["total_points"].median()),
        "mean": float(all_df["total_points"].mean()),
        "max": float(all_df["total_points"].max()),
        "p90": float(all_df["total_points"].quantile(0.9)),
    },
    "n_features_used": {
        p: len([c for c in selected_features[str(CODE_FROM_NAME[p])] if c in eval_df.columns])
        for p in POSITION_MAP.values()
    },
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote metrics to {out_path}")
print(f"Wrote charts to {IMG_DIR}")
