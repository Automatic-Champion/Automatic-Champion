"""
build_data_v7.py
Build train_v7.csv and test_v7.csv for V7 weekly points model.

Train: 2016-17 through 2023-24
Test:  2024-25

Key design decisions:
- position/team fixed for pre-2020-21 seasons via players_raw.csv
- opponent_team converted from numeric ID to team name
- starts derived as minutes > 0 for seasons before 2022-23
- xP left as NaN for pre-2020-21, xG stats left as NaN for pre-2022-23
- Rolling features computed with shift(1) — zero leakage
- future_points = next GW total_points (last GW of each season dropped)
"""

import os
import numpy as np
import pandas as pd

BASE     = os.path.join(os.path.dirname(__file__), '..', '..', 'Base Data')
OUT_DIR  = os.path.dirname(__file__)

TRAIN_SEASONS = ['2016-17', '2017-18', '2018-19', '2019-20',
                 '2020-21', '2021-22', '2022-23', '2023-24']
TEST_SEASON   = '2024-25'

# Columns to compute rolling features for
ROLL_COLS = [
    'total_points', 'minutes', 'ict_index', 'bps', 'influence',
    'creativity', 'threat', 'selected', 'value', 'transfers_balance',
    'xP', 'expected_goals', 'expected_assists',
    'expected_goal_involvements', 'expected_goals_conceded',
    'saves', 'clean_sheets', 'goals_scored', 'assists', 'bonus',
    'starts', 'team_goals_scored', 'team_goals_conceded',
]

POS_MAP = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}


# ── loaders ──────────────────────────────────────────────────────────────────

def load_season(season):
    path = os.path.join(BASE, season, 'gws', 'merged_gw.csv')
    df = pd.read_csv(path, encoding='latin1')
    df['season'] = season
    return df


def load_master_team_list():
    path = os.path.join(BASE, 'master_team_list.csv')
    return pd.read_csv(path, encoding='latin1')


# ── fix position + team for pre-2020-21 ──────────────────────────────────────

def fix_position_and_team(df, season, master):
    """
    For seasons where position/team are missing, join from players_raw.csv.
    players_raw has: id (= element), element_type (1-4), team (numeric ID)
    master_team_list maps numeric ID -> team name per season.
    """
    needs_fix = 'position' not in df.columns or df['position'].isna().all()
    if not needs_fix:
        return df

    raw_path = os.path.join(BASE, season, 'players_raw.csv')
    raw = pd.read_csv(raw_path, encoding='latin1', usecols=['id', 'element_type', 'team'])
    raw = raw.rename(columns={'id': 'element', 'team': 'team_id'})
    raw['position'] = raw['element_type'].map(POS_MAP)
    raw = raw[['element', 'position', 'team_id']].drop_duplicates('element')

    df = df.merge(raw, on='element', how='left')

    season_teams = master[master['season'] == season][['team', 'team_name']]
    season_teams = season_teams.rename(columns={'team': 'team_id', 'team_name': 'team'})
    df = df.merge(season_teams, on='team_id', how='left')
    df = df.drop(columns=['team_id', 'element_type'], errors='ignore')

    return df


# ── resolve opponent_team ID -> name ─────────────────────────────────────────

def build_opponent_map(df, season, master):
    """
    Build {opponent_team_id: team_name} for a season.
    For seasons in master_team_list: use directly.
    For 2024-25 (not in master): derive from the data itself by
    matching home/away players within each fixture.
    """
    master_season = master[master['season'] == season]
    if len(master_season) > 0:
        return dict(zip(master_season['team'].astype(int),
                        master_season['team_name']))

    # Derive from data: for each fixture, home players face away team and vice versa
    mapping = {}
    if 'team' not in df.columns:
        return mapping

    for _, fdf in df.groupby('fixture'):
        home = fdf[fdf['was_home'] == True]
        away = fdf[fdf['was_home'] == False]
        if len(home) == 0 or len(away) == 0:
            continue
        away_name = away['team'].iloc[0]
        away_id   = int(home['opponent_team'].iloc[0])
        home_name = home['team'].iloc[0]
        home_id   = int(away['opponent_team'].iloc[0])
        mapping[away_id] = away_name
        mapping[home_id] = home_name

    return mapping


def resolve_opponent_team(df, opp_map):
    df['opponent_team'] = df['opponent_team'].map(
        lambda x: opp_map.get(int(x), str(x)) if pd.notna(x) else x
    )
    return df


# ── derive columns ────────────────────────────────────────────────────────────

def derive_columns(df):
    df['team_goals_scored']   = np.where(df['was_home'], df['team_h_score'], df['team_a_score'])
    df['team_goals_conceded'] = np.where(df['was_home'], df['team_a_score'], df['team_h_score'])

    if 'starts' not in df.columns:
        df['starts'] = (df['minutes'] > 0).astype(float)
    else:
        df['starts'] = df['starts'].astype(float)

    # Fill NaN scores (postponed/void matches e.g. 2019-20 COVID) with 0
    df['team_goals_scored']   = df['team_goals_scored'].fillna(0)
    df['team_goals_conceded'] = df['team_goals_conceded'].fillna(0)

    # Normalize position names (FPL used 'GKP' in some seasons instead of 'GK')
    if 'position' in df.columns:
        df['position'] = df['position'].replace('GKP', 'GK')

    # Drop manager rows — FPL added managers (AM = Assistant Manager) in 2024-25
    if 'position' in df.columns:
        df = df[~df['position'].isin(['AM', 'MNG'])].copy()

    return df


# ── rolling features ──────────────────────────────────────────────────────────

def compute_rolling(df):
    df = df.sort_values(['season', 'element', 'GW']).copy()

    for col in ROLL_COLS:
        if col not in df.columns:
            continue
        g = df.groupby(['season', 'element'])[col]
        df[f'{col}_last']  = g.transform(lambda x: x.shift(1))
        df[f'{col}_mean3'] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df[f'{col}_mean5'] = g.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        df[f'{col}_mean7'] = g.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())

    return df


# ── label ─────────────────────────────────────────────────────────────────────

def compute_label(df):
    df = df.sort_values(['season', 'element', 'GW']).copy()
    df['future_points'] = df.groupby(['season', 'element'])['total_points'].transform(
        lambda x: x.shift(-1)
    )
    last_gw = df.groupby('season')['GW'].transform('max')
    df = df[df['GW'] != last_gw].copy()
    return df


# ── column selection ──────────────────────────────────────────────────────────

KEEP_RAW = [
    'season', 'element', 'name', 'position', 'team', 'GW',
    'was_home', 'opponent_team',
    'minutes', 'total_points', 'value',
    'ict_index', 'influence', 'creativity', 'threat',
    'bps', 'bonus', 'goals_scored', 'assists',
    'clean_sheets', 'goals_conceded', 'saves',
    'selected', 'transfers_balance',
    'starts', 'team_goals_scored', 'team_goals_conceded',
    'xP',
    'expected_goals', 'expected_assists',
    'expected_goal_involvements', 'expected_goals_conceded',
]


def select_columns(df):
    roll_cols = [c for c in df.columns if any(
        c == f'{r}_{s}' for r in ROLL_COLS for s in ['last', 'mean3', 'mean5', 'mean7']
    )]
    keep = [c for c in KEEP_RAW if c in df.columns] + roll_cols + ['future_points']
    return df[keep]


# ── main ──────────────────────────────────────────────────────────────────────

def build_dataset(seasons):
    master = load_master_team_list()
    all_dfs = []

    for season in seasons:
        print(f'  Loading {season}...', end=' ')
        df = load_season(season)
        df = fix_position_and_team(df, season, master)
        df = derive_columns(df)
        opp_map = build_opponent_map(df, season, master)
        df = resolve_opponent_team(df, opp_map)
        all_dfs.append(df)
        print(f'{len(df):,} rows')

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f'  Combined: {len(combined):,} rows — computing rolling features...')
    combined = compute_rolling(combined)
    print(f'  Computing labels...')
    combined = compute_label(combined)
    combined = select_columns(combined)
    return combined


if __name__ == '__main__':
    print('=== Building train (2016-17 → 2023-24) ===')
    train = build_dataset(TRAIN_SEASONS)
    train_path = os.path.join(OUT_DIR, 'train_v7.csv')
    train.to_csv(train_path, index=False)
    print(f'Saved: {train_path}  ({len(train):,} rows, {len(train.columns)} cols)\n')

    print('=== Building test (2024-25) ===')
    test = build_dataset([TEST_SEASON])
    test_path = os.path.join(OUT_DIR, 'test_v7.csv')
    test.to_csv(test_path, index=False)
    print(f'Saved: {test_path}  ({len(test):,} rows, {len(test.columns)} cols)\n')

    print('=== Summary ===')
    print(f'Train shape: {train.shape}')
    print(f'Test shape:  {test.shape}')
    print(f'\nTrain seasons: {sorted(train["season"].unique())}')
    print(f'Test seasons:  {sorted(test["season"].unique())}')
    print(f'\nColumns with NaN (train):')
    nan_cols = train.isna().sum()
    nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)
    print(nan_cols.to_string())
