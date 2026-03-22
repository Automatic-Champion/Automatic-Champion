from __future__ import annotations

from pathlib import Path

import pandas as pd

from training.xgboost_position_models import main

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "historical_exports"

SEASONS_TRAIN = ["2019-20", "2020-21", "2021-22", "2022-23"]
SEASON_TEST = "2024-25"


def load_season(season: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"historical_players_{season}.csv")


def run() -> None:
    df_train = pd.concat(
        [load_season(season) for season in SEASONS_TRAIN], ignore_index=True
    )
    df_test = load_season(SEASON_TEST)
    main(df_train, df_test)


if __name__ == "__main__":
    run()
