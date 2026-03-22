from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "historical_exports"
OUTPUT_PATH = BASE_DIR / "visuals" / "position_correlation_comparison.png"
HEATMAP_PATH = BASE_DIR / "visuals" / "position_top_features_heatmap.png"
MULTIPLES_PATH = BASE_DIR / "visuals" / "position_top_features_multiples.png"
LOLLIPOP_PATH = BASE_DIR / "visuals" / "position_top_features_lollipop.png"

SEASONS_TRAIN = ["2019-20", "2020-21", "2021-22", "2022-23"]

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def load_season(season: str) -> pd.DataFrame:
    path = DATA_DIR / f"historical_players_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def top_correlations(df: pd.DataFrame, target: str, top_n: int = 10) -> pd.Series:
    numeric = df.select_dtypes(include="number")
    if target not in numeric.columns:
        raise ValueError(f"Target '{target}' not found in numeric columns.")
    corr = numeric.corr()[target].drop(labels=[target]).dropna()
    return corr.abs().sort_values(ascending=False).head(top_n)


def main() -> None:
    frames = [load_season(season) for season in SEASONS_TRAIN]
    df_train = pd.concat(frames, ignore_index=True)

    if "element_type" not in df_train.columns:
        raise ValueError("Expected 'element_type' column in training data.")

    df_train["position_name"] = df_train["element_type"].map(POSITION_MAP)

    top_by_position = {}
    for position in POSITION_MAP.values():
        subset = df_train[df_train["position_name"] == position]
        if subset.empty:
            print(f"\n{position}: no rows")
            continue
        top = top_correlations(subset, target="total_points", top_n=10)
        top_by_position[position] = top
        print(f"\nTop 10 correlations for {position}:")
        for feature, value in top.items():
            print(f"{feature}: {value:.3f}")

    feature_candidates = [
        "1_years_past_clean_sheets",
        "1_years_past_threat",
        "1_years_past_creativity",
    ]
    fallback_threat = "1_years_past_goals_scored"
    features = []
    for feat in feature_candidates:
        if feat in df_train.columns:
            features.append(feat)
        elif feat == "1_years_past_threat" and fallback_threat in df_train.columns:
            features.append(fallback_threat)
        else:
            raise ValueError(f"Required feature missing: {feat}")

    corr_by_position = {pos: [] for pos in POSITION_MAP.values()}
    for pos in POSITION_MAP.values():
        subset = df_train[df_train["position_name"] == pos]
        numeric = subset.select_dtypes(include="number")
        corr = numeric.corr()["total_points"]
        for feat in features:
            corr_by_position[pos].append(corr.get(feat, float("nan")))

    positions = list(POSITION_MAP.values())
    x = range(len(positions))
    width = 0.25

    plt.figure(figsize=(10, 6))
    for idx, feat in enumerate(features):
        offsets = [i + (idx - 1) * width for i in x]
        values = [corr_by_position[pos][idx] for pos in positions]
        label = feat.replace("1_years_past_", "").replace("_", " ").title()
        plt.bar(offsets, values, width=width, label=label)

    plt.xticks(list(x), positions)
    plt.ylabel("Correlation with Total Points")
    plt.xlabel("Position")
    plt.title("Feature Correlation by Position")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved plot to: {OUTPUT_PATH}")

    if top_by_position:
        all_features = sorted(
            {feature for series in top_by_position.values() for feature in series.index}
        )
        heatmap_data = pd.DataFrame(index=all_features, columns=POSITION_MAP.values())
        for pos, series in top_by_position.items():
            for feature in all_features:
                heatmap_data.at[feature, pos] = series.get(feature, 0.0)
        heatmap_data = heatmap_data.astype(float)

        plt.figure(figsize=(10, max(6, len(all_features) * 0.25)))
        plt.imshow(heatmap_data.values, aspect="auto", cmap="viridis")
        plt.colorbar(label="Abs Correlation")
        plt.yticks(range(len(all_features)), all_features)
        plt.xticks(range(len(POSITION_MAP)), list(POSITION_MAP.values()))
        plt.title("Top Feature Correlations by Position")
        plt.tight_layout()
        plt.savefig(HEATMAP_PATH, dpi=150)
        print(f"Saved plot to: {HEATMAP_PATH}")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes = axes.ravel()
        for idx, pos in enumerate(POSITION_MAP.values()):
            ax = axes[idx]
            series = top_by_position.get(pos)
            if series is None or series.empty:
                ax.set_title(f"{pos} (no data)")
                ax.axis("off")
                continue
            series = series.sort_values(ascending=True)
            ax.barh(series.index, series.values, color="#4C78A8")
            ax.set_title(f"{pos} Top Features")
            ax.set_xlabel("Abs Correlation")
        plt.savefig(MULTIPLES_PATH, dpi=150)
        print(f"Saved plot to: {MULTIPLES_PATH}")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes = axes.ravel()
        for idx, pos in enumerate(POSITION_MAP.values()):
            ax = axes[idx]
            series = top_by_position.get(pos)
            if series is None or series.empty:
                ax.set_title(f"{pos} (no data)")
                ax.axis("off")
                continue
            series = series.sort_values(ascending=True)
            y = range(len(series))
            ax.hlines(y, [0] * len(series), series.values, color="#9EC1CF", linewidth=2)
            ax.plot(series.values, y, "o", color="#1F77B4")
            ax.set_yticks(y)
            ax.set_yticklabels(series.index)
            ax.set_title(f"{pos} Top Features")
            ax.set_xlabel("Abs Correlation")
        plt.savefig(LOLLIPOP_PATH, dpi=150)
        print(f"Saved plot to: {LOLLIPOP_PATH}")


if __name__ == "__main__":
    main()
