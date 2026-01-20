from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Base Data"
OUTPUT_DIR = BASE_DIR / "historical_player_exports" / "output"

SEASONS = [
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

TARGET_SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

POSITION_MAP = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD",
}
POSITION_CODE_MAP = {value: key for key, value in POSITION_MAP.items()}
INTEGER_SUFFIXES = (
    "assists",
    "bonus",
    "bps",
    "clean_sheets",
    "goals_conceded",
    "goals_scored",
    "minutes",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "red_cards",
    "saves",
    "total_points",
    "yellow_cards",
)
INTEGER_COLS = {"games_played", "gw_games_played"}

def normalize_name(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    if "_" in text:
        parts = text.split("_")
        if parts and parts[-1].isdigit():
            parts = parts[:-1]
        text = " ".join(parts)
    text = " ".join(text.split())
    return text


def normalize_element_type(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    if text in POSITION_CODE_MAP:
        return POSITION_CODE_MAP[text]
    try:
        number = int(float(text))
    except (TypeError, ValueError):
        return pd.NA
    return number if number in POSITION_MAP else pd.NA


def load_player_positions(season):
    path = DATA_DIR / season / "players_raw.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="latin-1")
    df["name_key"] = (
        df["first_name"].astype(str).str.strip()
        + " "
        + df["second_name"].astype(str).str.strip()
    ).map(normalize_name)
    df["element_type"] = pd.to_numeric(df.get("element_type"), errors="coerce")
    return df.dropna(subset=["name_key"]).set_index("name_key")["element_type"].to_dict()


def load_cleaned_players(season):
    path = DATA_DIR / season / "cleaned_players.csv"
    df = pd.read_csv(path, encoding="latin-1")
    df["name_key"] = (
        df["first_name"].astype(str).str.strip()
        + " "
        + df["second_name"].astype(str).str.strip()
    ).map(normalize_name)
    if "element_type" in df.columns:
        df["element_type"] = df["element_type"].map(normalize_element_type)
    else:
        df["element_type"] = pd.NA
    if df["element_type"].isna().all():
        position_map = load_player_positions(season)
        df["element_type"] = df["name_key"].map(position_map)
    return df


def load_gw_data(season):
    gw_dir = DATA_DIR / season / "gws"
    if not gw_dir.exists():
        return pd.DataFrame()
    files = sorted(gw_dir.glob("gw*.csv"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(path, encoding="latin-1") for path in files]
    df = pd.concat(frames, ignore_index=True)
    df["name_key"] = df["name"].map(normalize_name)
    return df


def gw_aggregates(season):
    df = load_gw_data(season)
    if df.empty:
        empty = pd.DataFrame(columns=["name_key"])
        return empty, {}

    for col in ["minutes", "penalties_missed", "penalties_saved", "own_goals", "saves"]:
        if col not in df.columns:
            df[col] = pd.NA

    numeric_cols = ["minutes", "penalties_missed", "penalties_saved", "own_goals", "saves"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    games_played = df["minutes"].gt(0).groupby(df["name_key"]).sum()
    total_minutes = df.groupby("name_key")["minutes"].sum()
    minutes_per_game = total_minutes / games_played.replace(0, pd.NA)

    penalties_missed = df.groupby("name_key")["penalties_missed"].sum().reindex(games_played.index)
    penalties_saved = df.groupby("name_key")["penalties_saved"].sum().reindex(games_played.index)
    own_goals = df.groupby("name_key")["own_goals"].sum().reindex(games_played.index)
    saves = df.groupby("name_key")["saves"].sum().reindex(games_played.index)

    agg = pd.DataFrame(
        {
            "name_key": games_played.index,
            "games_played": games_played.values,
            "minutes_per_game": minutes_per_game.reindex(games_played.index).values,
            "penalties_missed": penalties_missed.values,
            "penalties_saved": penalties_saved.values,
            "own_goals": own_goals.values,
            "saves": saves.values,
        }
    )

    position_map = {}
    if "position" in df.columns:
        position_series = df.groupby("name_key")["position"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA
        )
        position_map = position_series.to_dict()
    return agg, position_map


def add_prefixed_columns(df, prefix):
    prefixed = df.add_prefix(prefix)
    return prefixed


def impute_missing_stats(df):
    name_cols = {
        "first_name",
        "second_name",
        "name_key",
        "position",
        "element_type",
    }
    element_type_cols = {col for col in df.columns if col.endswith("element_type")}
    name_cols.update(element_type_cols)
    for col in df.columns:
        if col.endswith("_first_name") or col.endswith("_second_name"):
            name_cols.add(col)

    numeric_cols = [col for col in df.columns if col not in name_cols]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    overall_means = df[numeric_cols].mean()
    position_means = df.groupby("position")[numeric_cols].mean()

    for idx, row in df.iterrows():
        position = row.get("position", pd.NA)
        price_now = row.get("price_now", pd.NA)
        for col in numeric_cols:
            if pd.notna(row[col]):
                continue
            if pd.notna(position) and pd.notna(price_now):
                mask = (df["position"] == position) & df["price_now"].between(
                    price_now - 5, price_now + 5
                )
                candidate_mean = df.loc[mask, col].mean()
                if pd.notna(candidate_mean):
                    df.at[idx, col] = candidate_mean
                    continue
            if pd.notna(position) and position in position_means.index:
                pos_mean = position_means.at[position, col]
                if pd.notna(pos_mean):
                    df.at[idx, col] = pos_mean
                    continue
            df.at[idx, col] = overall_means[col]

    return df


def round_numeric(df, decimals=1):
    name_cols = {"first_name", "second_name", "element_type"}
    numeric_cols = [col for col in df.columns if col not in name_cols]
    for col in numeric_cols:
        if col in INTEGER_COLS or col.endswith(INTEGER_SUFFIXES):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").round(decimals)
    return df


def enforce_integer_columns(df):
    for col in df.columns:
        if col in INTEGER_COLS or col.endswith(INTEGER_SUFFIXES) or col.endswith("element_type"):
            df[col] = pd.to_numeric(df[col], errors="coerce").round(0).astype("Int64")
    return df


def missing_historical_report(df, seasons_back):
    report = {}
    for offset in range(1, seasons_back + 1):
        prefix = f"{offset}_years_past_"
        hist_cols = [col for col in df.columns if col.startswith(prefix)]
        if not hist_cols:
            report[offset] = {"missing": [], "present": []}
            continue
        missing_mask = df[hist_cols].isna().all(axis=1)
        missing_names = (
            df.loc[missing_mask, ["first_name", "second_name"]]
            .fillna("")
            .agg(" ".join, axis=1)
            .str.strip()
            .tolist()
        )
        present_names = (
            df.loc[~missing_mask, ["first_name", "second_name"]]
            .fillna("")
            .agg(" ".join, axis=1)
            .str.strip()
            .tolist()
        )
        report[offset] = {
            "missing": [name for name in missing_names if name],
            "present": [name for name in present_names if name],
        }
    return report


def season_index(season):
    return SEASONS.index(season)


def historical_seasons(target_season):
    idx = season_index(target_season)
    return list(reversed(SEASONS[idx - 3 : idx]))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for season in TARGET_SEASONS:
        base = load_cleaned_players(season)
        base = base.rename(columns={"now_cost": "price_now"})
        base["price_now"] = pd.to_numeric(base.get("price_now"), errors="coerce")

        gw_base_agg, gw_position_map = gw_aggregates(season)
        base_positions = base.get("element_type")
        if base_positions is not None:
            base_positions = base_positions.map(POSITION_MAP)
        base["position"] = base_positions
        if gw_position_map:
            base["position"] = base["position"].fillna(base["name_key"].map(gw_position_map))

        output = base[
            [
                "first_name",
                "second_name",
                "total_points",
                "price_now",
                "element_type",
                "name_key",
                "position",
            ]
        ].copy()

        for offset, hist_season in enumerate(historical_seasons(season), start=1):
            hist_cleaned = load_cleaned_players(hist_season)
            hist_cleaned = hist_cleaned.drop(columns=["first_name", "second_name"], errors="ignore")
            hist_pref = add_prefixed_columns(hist_cleaned, f"{offset}_years_past_")
            output = output.merge(
                hist_pref,
                left_on="name_key",
                right_on=f"{offset}_years_past_name_key",
                how="left",
            ).drop(columns=[f"{offset}_years_past_name_key"])

            gw_agg, _ = gw_aggregates(hist_season)
            gw_pref = add_prefixed_columns(gw_agg, f"{offset}_years_past_gw_")
            output = output.merge(
                gw_pref,
                left_on="name_key",
                right_on=f"{offset}_years_past_gw_name_key",
                how="left",
            ).drop(columns=[f"{offset}_years_past_gw_name_key"])

        missing_report = missing_historical_report(output, 3)
        output = impute_missing_stats(output)
        output = round_numeric(output, decimals=1)
        output = enforce_integer_columns(output)
        output = output.drop(columns=["name_key"])

        out_path = OUTPUT_DIR / f"historical_players_{season}.csv"
        output.to_csv(out_path, index=False)

        report_path = OUTPUT_DIR / f"historical_players_{season}_missing.md"
        with report_path.open("w", encoding="utf-8") as handle:
            handle.write(f"# Missing historical data for {season}\n\n")
            for offset in range(1, 4):
                entry = missing_report.get(offset, {"missing": [], "present": []})
                missing_names = entry["missing"]
                present_names = entry["present"]
                handle.write(f"## {offset} years past\n")
                handle.write(f"Has data: {len(present_names)}\n")
                handle.write(f"Missing data: {len(missing_names)}\n\n")
                for name in missing_names:
                    handle.write(f"- {name}\n")
                handle.write("\n")


if __name__ == "__main__":
    main()
