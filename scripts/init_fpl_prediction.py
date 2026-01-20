from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "historical_player_exports" / "output"

SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]


def load_season(season: str) -> pd.DataFrame:
    path = DATA_DIR / f"historical_players_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def find_position_column(df: pd.DataFrame) -> tuple[str, list] | tuple[None, list]:
    candidates = []
    direct_candidates = [
        "element_type",
        "position",
        "pos",
        "player_position",
        "current_position",
    ]
    for name in direct_candidates:
        if name in df.columns:
            series = df[name].replace("", pd.NA)
            unique_vals = sorted(series.dropna().unique().tolist())
            return name, unique_vals

    for col in df.columns:
        if col.endswith("element_type") or col.endswith("position") or col.endswith("_pos"):
            if col not in candidates:
                candidates.append(col)

    scored = []
    for col in candidates:
        series = df[col].replace("", pd.NA)
        non_null = series.notna().sum()
        unique_vals = sorted(series.dropna().unique().tolist())
        scored.append((non_null, len(unique_vals), col, unique_vals))

    if not scored:
        return None, []

    scored.sort(reverse=True)
    _, _, best_col, unique_vals = scored[0]
    return best_col, unique_vals


def main() -> None:
    frames = {season: load_season(season) for season in SEASONS}

    df_train = pd.concat(
        [frames[season] for season in ["2019-20", "2020-21", "2021-22", "2022-23"]],
        ignore_index=True,
    )
    df_val = frames["2023-24"].copy()
    df_test = frames["2024-25"].copy()

    position_col, unique_positions = find_position_column(df_train)
    if position_col is None:
        raise ValueError(
            "No position column found. Expected columns like 'position', "
            "'element_type', or a prefixed '*_element_type'."
        )

    print(f"df_train shape: {df_train.shape}")
    print(f"df_val shape: {df_val.shape}")
    print(f"df_test shape: {df_test.shape}")
    print(f"position column: {position_col}")
    print(f"unique positions: {unique_positions}")


if __name__ == "__main__":
    main()
