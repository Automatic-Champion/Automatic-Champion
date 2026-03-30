# ML Model Report — Automatic Champion

**Generated:** 2026-03-29
**Purpose:** Comprehensive documentation of all ML models, training data, experiments, and baselines for anyone building or comparing new models.

---

## Section 1: Training Data

### Source Files

All training data lives in `data/historical_exports/`. Each season has its own CSV:

| File | Season | Rows (excl. header) |
|------|--------|---------------------|
| `data/historical_exports/historical_players_2019-20.csv` | 2019-20 | 667 |
| `data/historical_exports/historical_players_2020-21.csv` | 2020-21 | 714 |
| `data/historical_exports/historical_players_2021-22.csv` | 2021-22 | 740 |
| `data/historical_exports/historical_players_2022-23.csv` | 2022-23 | 784 |
| `data/historical_exports/historical_players_2023-24.csv` | 2023-24 | 872 |
| `data/historical_exports/historical_players_2024-25.csv` | 2024-25 | 807 |

The inference-time player pool is `data/players_merged_2024-25.csv` (807 rows). This file has identical columns to the historical exports plus `id` and `team_name`.

### Available Seasons

6 seasons: 2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25.

### Column Inventory

The historical CSVs and `players_merged_2024-25.csv` share this schema (the merged file adds `id` and `team_name`):

**Demographics / identifiers:**
- `first_name`, `second_name`
- `total_points` (TARGET VARIABLE)
- `price_now` (current season price, in tenths of £M — e.g., 54 = £5.4M)
- `element_type` (1=GK, 2=DEF, 3=MID, 4=FWD)
- `position` (string version, may be empty in historical files)

**1-year lag features (`1_years_past_*`):**
- `1_years_past_goals_scored`
- `1_years_past_assists`
- `1_years_past_total_points`
- `1_years_past_minutes`
- `1_years_past_goals_conceded`
- `1_years_past_creativity`
- `1_years_past_influence`
- `1_years_past_threat`
- `1_years_past_bonus`
- `1_years_past_bps`
- `1_years_past_ict_index`
- `1_years_past_clean_sheets`
- `1_years_past_red_cards`
- `1_years_past_yellow_cards`
- `1_years_past_selected_by_percent`
- `1_years_past_now_cost`
- `1_years_past_element_type`
- `1_years_past_gw_games_played`
- `1_years_past_gw_minutes_per_game`
- `1_years_past_gw_penalties_missed`
- `1_years_past_gw_penalties_saved`
- `1_years_past_gw_own_goals`
- `1_years_past_gw_saves`

**2-year lag features (`2_years_past_*`):** Same 23 stat columns as above, prefixed with `2_years_past_`.

**3-year lag features (`3_years_past_*`):** Same 23 stat columns as above, prefixed with `3_years_past_`.

Total: 6 demographic/id columns + 23 × 3 lag columns = 75 columns in historical files, 77 in merged file.

### Target Variable

`total_points` — the player's FPL total points for the current season. This is a continuous integer value. Range varies from -1 to 244 in the 2023-24 test set.

### Missing Values

Each season has a corresponding `*_missing.md` file documenting players with no prior-season data. For example, `historical_players_2023-24_missing.md` reports:
- **519 players** have 1-year-past data
- **353 players** are missing 1-year-past data (new signings, youth players, promoted team players)

Players missing lag data have empty/NaN values for those columns. All training scripts handle this by filling NaN with 0 (`fillna(0)`) or using `SimpleImputer(strategy="median")`.

Missing data is more severe for deeper lags (2-year, 3-year) since players must have been in FPL for multiple consecutive seasons.

---

## Section 2: Feature Engineering

### Features Used by Deployed Models

The function `_build_feature_cols()` in `src/team_builder.py:58-81` selects:

1. `price_now`
2. All columns matching `1_years_past_*` that are numeric

This produces 24 features total (price_now + 23 one-year lag stats). The exact list:
- `price_now`
- `1_years_past_goals_scored`
- `1_years_past_assists`
- `1_years_past_total_points`
- `1_years_past_minutes`
- `1_years_past_goals_conceded`
- `1_years_past_creativity`
- `1_years_past_influence`
- `1_years_past_threat`
- `1_years_past_bonus`
- `1_years_past_bps`
- `1_years_past_ict_index`
- `1_years_past_clean_sheets`
- `1_years_past_red_cards`
- `1_years_past_yellow_cards`
- `1_years_past_selected_by_percent`
- `1_years_past_now_cost`
- `1_years_past_element_type`
- `1_years_past_gw_games_played`
- `1_years_past_gw_minutes_per_game`
- `1_years_past_gw_penalties_missed`
- `1_years_past_gw_penalties_saved`
- `1_years_past_gw_own_goals`
- `1_years_past_gw_saves`

### Derived Features

The **basic models** and **comparison models** (`train_position_models.py`, `train_and_compare_models.py`) use **no derived features** — raw columns only.

The **advanced models** (`train_advanced_models.py`) add 2 per-90 features computed in `add_per_90_features()`:
- `1_years_past_total_points_per_90` = (`1_years_past_total_points` / `1_years_past_minutes`) × 90
- `1_years_past_ict_index_per_90` = (`1_years_past_ict_index` / `1_years_past_minutes`) × 90

Both fill with 0.0 when minutes = 0.

### Features Available but NOT Used

- All `2_years_past_*` columns (23 features) — 2-year historical lag
- All `3_years_past_*` columns (23 features) — 3-year historical lag
- `1_years_past_element_type` is included by the column filter but is questionable (it's the position code, not a performance stat)

### How `price_now` is Handled

`price_now` is used **raw** — no scaling, normalization, or transformation. It is in FPL's native unit (tenths of £M). RandomForest and tree-based models are invariant to monotonic transformations, so this doesn't affect them. The advanced Ridge model uses `StandardScaler` in its pipeline.

### Feature Selection Steps

None. All numeric `1_years_past_*` columns plus `price_now` are used. No correlation filtering, no importance thresholding, no dimensionality reduction.

---

## Section 3: Currently Deployed Models (`position_model_*.joblib`)

### Training Configuration

- **Script:** `training/train_position_models.py`
- **Algorithm:** `RandomForestRegressor(n_estimators=100, random_state=42)`
- **All default sklearn hyperparameters** (max_depth=None, min_samples_split=2, min_samples_leaf=1, etc.)
- **Training seasons:** 2019-20, 2020-21, 2021-22, 2022-23
- **Test season:** 2023-24 (holdout, time-based split)
- **Missing value handling:** `fillna(0)`
- **No cross-validation** — single train/test split
- **No hyperparameter tuning**

### Model Files

| Position | File | Element Type | File Size |
|----------|------|-------------|-----------|
| GK | `models/position_model_1.joblib` | 1 | 1,806,049 bytes (1.7 MB) |
| DEF | `models/position_model_2.joblib` | 2 | 7,026,913 bytes (6.7 MB) |
| MID | `models/position_model_3.joblib` | 3 | 8,892,433 bytes (8.5 MB) |
| FWD | `models/position_model_4.joblib` | 4 | 2,699,281 bytes (2.6 MB) |

### Metrics

The `train_position_models.py` script only outputs MAE. Full metrics come from `train_and_compare_models.py` which runs the same RF configuration (but with hyperparameter search). The closest comparable results from `visuals/model_metrics_by_position.csv` for RandomForest:

| Position | Train Seasons | Train Rows | Test Season | Test Rows | CV MAE | Test MAE | Test RMSE | Best Params |
|----------|--------------|------------|-------------|-----------|--------|----------|-----------|-------------|
| GK | 2019-20 to 2022-23 | 324 | 2023-24 | 100 | 19.631 | 14.617 | 28.075 | n_estimators=400, min_samples_leaf=1, max_depth=None |
| DEF | 2019-20 to 2022-23 | 987 | 2023-24 | 285 | 24.838 | 23.833 | 28.777 | n_estimators=400, min_samples_leaf=4, max_depth=None |
| MID | 2019-20 to 2022-23 | 1219 | 2023-24 | 374 | 24.746 | 24.849 | 36.190 | n_estimators=200, min_samples_leaf=2, max_depth=10 |
| FWD | 2019-20 to 2022-23 | 375 | 2023-24 | 113 | 29.229 | 27.496 | 44.605 | n_estimators=400, min_samples_leaf=4, max_depth=None |

**Note:** The deployed models use `n_estimators=100` with all defaults, while the comparison study searched over `{200, 400}` estimators and other params. The deployed models likely have slightly different (possibly worse) metrics than the table above.

### Top 10 Features (from `visuals/winner_top_features.csv`)

**GK** (winner: RandomForest):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | price_now | 0.4546 |
| 2 | 1_years_past_total_points | 0.1169 |
| 3 | 1_years_past_gw_saves | 0.0697 |
| 4 | 1_years_past_bps | 0.0517 |
| 5 | 1_years_past_selected_by_percent | 0.0425 |
| 6 | 1_years_past_gw_games_played | 0.0418 |
| 7 | 1_years_past_influence | 0.0368 |
| 8 | 1_years_past_now_cost | 0.0298 |
| 9 | 1_years_past_clean_sheets | 0.0239 |
| 10 | 1_years_past_ict_index | 0.0226 |

**DEF** (winner: XGBoost — note: this is from the comparison, not the deployed model):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | 1_years_past_clean_sheets | 0.2021 |
| 2 | 1_years_past_bps | 0.1463 |
| 3 | price_now | 0.1103 |
| 4 | 1_years_past_ict_index | 0.0950 |
| 5 | 1_years_past_influence | 0.0740 |
| 6 | 1_years_past_total_points | 0.0700 |
| 7 | 1_years_past_minutes | 0.0482 |
| 8 | 1_years_past_gw_games_played | 0.0389 |
| 9 | 1_years_past_goals_conceded | 0.0340 |
| 10 | 1_years_past_now_cost | 0.0275 |

**MID** (winner: RandomForest):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | 1_years_past_ict_index | 0.3631 |
| 2 | price_now | 0.1951 |
| 3 | 1_years_past_influence | 0.0964 |
| 4 | 1_years_past_total_points | 0.0631 |
| 5 | 1_years_past_now_cost | 0.0369 |
| 6 | 1_years_past_threat | 0.0316 |
| 7 | 1_years_past_gw_minutes_per_game | 0.0307 |
| 8 | 1_years_past_creativity | 0.0280 |
| 9 | 1_years_past_minutes | 0.0269 |
| 10 | 1_years_past_gw_games_played | 0.0240 |

**FWD** (winner: RandomForest):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | 1_years_past_creativity | 0.4148 |
| 2 | price_now | 0.1804 |
| 3 | 1_years_past_bps | 0.0633 |
| 4 | 1_years_past_gw_minutes_per_game | 0.0529 |
| 5 | 1_years_past_influence | 0.0491 |
| 6 | 1_years_past_ict_index | 0.0282 |
| 7 | 1_years_past_now_cost | 0.0269 |
| 8 | 1_years_past_total_points | 0.0262 |
| 9 | 1_years_past_threat | 0.0262 |
| 10 | 1_years_past_gw_games_played | 0.0223 |

### Prediction Analysis (from `visuals/winner_predictions.csv`)

**GK** (100 test players):
- Actual range: -1 to 153
- Predicted range: 0.145 to 135.975
- Many players with actual=0 (non-playing GKs) get predictions between 0.1 and 130.6 — notable overprediction for backup GKs who had prior-season data
- High scorers (100+) are generally predicted in the 77–136 range — reasonable but with variance

**DEF** (285 test players):
- Actual range: 0 to 182
- Predicted range: 4.53 to 172.55
- Players with actual=0 often predicted at 5–70 (significant overprediction for non-playing defenders)
- Top performers (120+) predicted in 70–172 range

**MID** (374 test players):
- Actual range: 0 to 244
- Predicted range: 1.86 to 226.74
- Extreme outlier: actual=244 predicted at 53.6 (massive underprediction of Salah-level seasons)
- actual=226 predicted at 95.6 (also significant underprediction)
- actual=211 predicted at 226.7 (close match for this one case)

**FWD** (113 test players):
- Actual range: 0 to 228
- Predicted range: 0.79 to 199.39
- actual=228 predicted at 120.8 (significant underprediction of breakout season)
- actual=217 predicted at 186.0 (closer)
- actual=0 predicted at 199.4 (massive overprediction — likely a player with strong prior stats who didn't play)

### Error Patterns

1. **Overprediction of non-playing players:** Players with actual=0 but prior-season stats get substantial predicted points. The model cannot distinguish "will not play this season" from "will play."
2. **Underprediction of exceptional seasons:** Players scoring 180+ points are consistently underpredicted. The model regresses toward the mean.
3. **Backup/rotation players:** The model struggles with players who had significant prior minutes but become bench players.

---

## Section 4: Advanced Model Experiments

### Training Configuration

- **Script:** `training/train_advanced_models.py`
- **Algorithms tested:** Ridge, RandomForest, XGBoost, LightGBM
- **Hyperparameter search:** 3-fold CV, GridSearchCV for Ridge, RandomizedSearchCV (5 iterations) for tree models
- **Sample weighting:** Tree models (RF, XGB, LGBM) use `sample_weight = total_points.clip(lower=0) + 1.0` to emphasize high-scoring players
- **Ridge does NOT use sample weights**
- **Per-90 features:** 2 additional features (`1_years_past_total_points_per_90`, `1_years_past_ict_index_per_90`)
- **Training seasons:** All available seasons before holdout (auto-discovered)
- **Holdout season:** 2023-24
- **Missing value handling:** `SimpleImputer(strategy="median")`

### Hyperparameter Search Spaces

**Ridge:** GridSearchCV
- `alpha`: [0.1, 1.0, 10.0, 25.0]

**RandomForest:** RandomizedSearchCV (5 iterations)
- `n_estimators`: [200, 400]
- `max_depth`: [None, 10, 20]
- `min_samples_leaf`: [1, 2, 4]

**XGBoost:** RandomizedSearchCV (5 iterations)
- `n_estimators`: [250, 500]
- `max_depth`: [2, 4, 6]
- `learning_rate`: [0.03, 0.08, 0.12]
- `subsample`: [0.8, 1.0]
- `colsample_bytree`: [0.8, 1.0]

**LightGBM:** RandomizedSearchCV (5 iterations)
- `n_estimators`: [250, 500]
- `num_leaves`: [15, 31, 63]
- `learning_rate`: [0.03, 0.08, 0.12]
- `subsample`: [0.8, 1.0]
- `colsample_bytree`: [0.8, 1.0]

### All Metrics (from `visuals/advanced_model_metrics_by_position.csv`)

| Position | Model | CV MAE | Test MAE | Test RMSE | Sample Weight | Best Params | Train Rows | Test Rows |
|----------|-------|--------|----------|-----------|---------------|-------------|------------|-----------|
| GK | Ridge | 23.879 | 17.000 | 27.466 | No | alpha=25.0 | 324 | 100 |
| GK | RandomForest | 19.164 | 15.168 | 28.776 | Yes | n_estimators=400, min_samples_leaf=1, max_depth=None | 324 | 100 |
| GK | XGBoost | 20.811 | 27.927 | 35.797 | Yes | subsample=0.8, n_estimators=250, max_depth=4, lr=0.12, colsample=0.8 | 324 | 100 |
| GK | LightGBM | 21.722 | 17.696 | 31.410 | Yes | subsample=0.8, num_leaves=31, n_estimators=250, lr=0.12, colsample=1.0 | 324 | 100 |
| DEF | Ridge | 25.316 | 21.004 | 26.030 | No | alpha=1.0 | 987 | 285 |
| DEF | RandomForest | 25.102 | 22.260 | 27.875 | Yes | n_estimators=400, min_samples_leaf=1, max_depth=None | 987 | 285 |
| DEF | XGBoost | 28.097 | 26.331 | 31.273 | Yes | subsample=0.8, n_estimators=500, max_depth=4, lr=0.08, colsample=0.8 | 987 | 285 |
| DEF | LightGBM | 28.734 | 27.649 | 33.507 | Yes | subsample=0.8, num_leaves=31, n_estimators=250, lr=0.12, colsample=1.0 | 987 | 285 |
| MID | Ridge | 26.353 | 26.624 | 36.719 | No | alpha=10.0 | 1219 | 374 |
| MID | RandomForest | 25.089 | 27.805 | 36.865 | Yes | n_estimators=400, min_samples_leaf=1, max_depth=None | 1219 | 374 |
| MID | XGBoost | 28.872 | 31.080 | 39.541 | Yes | subsample=0.8, n_estimators=500, max_depth=4, lr=0.08, colsample=0.8 | 1219 | 374 |
| MID | LightGBM | 29.188 | 30.431 | 38.455 | Yes | subsample=0.8, num_leaves=31, n_estimators=250, lr=0.12, colsample=1.0 | 1219 | 374 |
| FWD | Ridge | 28.168 | 31.622 | 46.523 | No | alpha=10.0 | 375 | 113 |
| FWD | RandomForest | 29.607 | 28.345 | 44.973 | Yes | n_estimators=200, min_samples_leaf=1, max_depth=None | 375 | 113 |
| FWD | XGBoost | 31.439 | 29.415 | 43.874 | Yes | subsample=0.8, n_estimators=500, max_depth=4, lr=0.08, colsample=0.8 | 375 | 113 |
| FWD | LightGBM | 33.498 | 32.753 | 48.668 | Yes | subsample=0.8, num_leaves=31, n_estimators=250, lr=0.12, colsample=1.0 | 375 | 113 |

### Winners Per Position

| Position | Winning Model | Test MAE |
|----------|--------------|----------|
| GK | RandomForest | 15.168 |
| DEF | Ridge | 21.004 |
| MID | Ridge | 26.624 |
| FWD | RandomForest | 28.345 |

### Top 10 Features — Advanced Winners (from `visuals/advanced_winner_top_features.csv`)

**GK** (RandomForest):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | price_now | 0.5856 |
| 2 | 1_years_past_ict_index_per_90 | 0.0704 |
| 3 | 1_years_past_selected_by_percent | 0.0497 |
| 4 | 1_years_past_now_cost | 0.0299 |
| 5 | 1_years_past_total_points_per_90 | 0.0277 |
| 6 | 1_years_past_total_points | 0.0241 |
| 7 | 1_years_past_goals_conceded | 0.0211 |
| 8 | 1_years_past_gw_saves | 0.0205 |
| 9 | 1_years_past_influence | 0.0199 |
| 10 | 1_years_past_creativity | 0.0189 |

**DEF** (Ridge — importance = absolute coefficient values, not comparable scale to RF):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | 1_years_past_minutes | 37.877 |
| 2 | price_now | 32.016 |
| 3 | 1_years_past_total_points | 24.648 |
| 4 | 1_years_past_bps | 16.570 |
| 5 | 1_years_past_goals_conceded | 8.302 |
| 6 | 1_years_past_gw_games_played | 8.217 |
| 7 | 1_years_past_now_cost | 8.015 |
| 8 | 1_years_past_clean_sheets | 6.467 |
| 9 | 1_years_past_assists | 4.710 |
| 10 | 1_years_past_yellow_cards | 4.285 |

**MID** (Ridge):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | price_now | 34.005 |
| 2 | 1_years_past_minutes | 13.522 |
| 3 | 1_years_past_now_cost | 11.180 |
| 4 | 1_years_past_goals_scored | 9.847 |
| 5 | 1_years_past_total_points | 8.126 |
| 6 | 1_years_past_bps | 7.959 |
| 7 | 1_years_past_assists | 7.608 |
| 8 | 1_years_past_threat | 7.509 |
| 9 | 1_years_past_goals_conceded | 6.047 |
| 10 | 1_years_past_selected_by_percent | 3.765 |

**FWD** (RandomForest):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | price_now | 0.3943 |
| 2 | 1_years_past_creativity | 0.0962 |
| 3 | 1_years_past_gw_minutes_per_game | 0.0507 |
| 4 | 1_years_past_total_points_per_90 | 0.0479 |
| 5 | 1_years_past_bps | 0.0432 |
| 6 | 1_years_past_bonus | 0.0379 |
| 7 | 1_years_past_now_cost | 0.0370 |
| 8 | 1_years_past_ict_index_per_90 | 0.0364 |
| 9 | 1_years_past_threat | 0.0347 |
| 10 | 1_years_past_ict_index | 0.0307 |

### Differences from Basic Models

1. **Per-90 features:** 2 additional engineered features (total_points_per_90, ict_index_per_90)
2. **Sample weighting:** Tree models weight high-scoring players more heavily
3. **Median imputation:** Instead of fillna(0)
4. **Hyperparameter tuning:** GridSearchCV/RandomizedSearchCV with 3-fold CV
5. **Pipeline architecture:** sklearn Pipelines with preprocessing steps

### Why DEF and MID Advanced Models Are Broken

`advanced_model_DEF.joblib` and `advanced_model_MID.joblib` are only 4,818 bytes each, compared to GK at 7.4 MB and FWD at 5.5 MB. The winners for DEF and MID were **Ridge** (a linear model). Ridge models are tiny because they store only a coefficient vector (24–26 floats) rather than hundreds of decision trees. These models are NOT broken — they are simply small because Ridge is a compact model. However, they perform worse than the basic RF models on DEF (Ridge test MAE 21.004 vs RF 23.833 — actually Ridge wins here) and MID (Ridge test MAE 26.624 vs RF 24.849 — Ridge loses).

**Important correction:** The DEF advanced model (Ridge, MAE 21.004) actually outperforms the basic RF comparison model (MAE 23.833). The MID advanced model (Ridge, MAE 26.624) slightly underperforms the basic RF (MAE 24.849). The sample weighting in the advanced experiment may have hurt tree model performance for DEF and MID, causing Ridge to win despite being a simpler model.

### Advanced vs Basic Comparison (Same Test Set: 2023-24)

| Position | Basic RF Test MAE | Advanced Winner | Advanced Test MAE | Better? |
|----------|-------------------|-----------------|-------------------|---------|
| GK | 14.617 | RandomForest | 15.168 | Basic RF wins |
| DEF | 23.833 | Ridge | 21.004 | Advanced wins |
| MID | 24.849 | Ridge | 26.624 | Basic RF wins |
| FWD | 27.496 | RandomForest | 28.345 | Basic RF wins |

The advanced experiment's sample weighting and hyperparameter choices did not improve results for 3 of 4 positions. Only DEF benefited (Ridge without sample weights).

---

## Section 5: XGBoost Experiments

### Script: `training/xgboost_position_models.py`

Called via `training/run_xgboost_positions.py`.

**Training seasons:** 2019-20, 2020-21, 2021-22, 2022-23
**Test season:** 2024-25 (different from other experiments which use 2023-24)

**Feature differences:** Adds `transfers_in` if present (in addition to `1_years_past_*` and `price_now`).

**Hyperparameters (hardcoded, per position):**

| Position | max_depth | learning_rate | n_estimators |
|----------|-----------|---------------|--------------|
| GK (1) | 4 | 0.1 | 200 |
| DEF (2) | 4 | 0.1 | 200 |
| MID (3) | 2 | 0.05 | 500 |
| FWD (4) | 2 | 0.05 | 500 |

All use `objective="reg:squarederror"`, `random_state=42`.

### Saved Metrics

No CSV output. The script prints MAE per position and a weighted average MAE to stdout. No saved model files. It also generates a GK feature importance plot at `outputs/gk_feature_importance.png`.

### Comparison

Since this experiment tests on 2024-25 while others test on 2023-24, the results are **not directly comparable**. The metrics from `model_metrics_by_position.csv` for XGBoost on the 2023-24 test set (from the comparison script) are:

| Position | XGBoost Test MAE (2023-24) | XGBoost Test RMSE (2023-24) |
|----------|---------------------------|----------------------------|
| GK | 15.448 | 28.939 |
| DEF | 20.507 | 26.514 |
| MID | 26.300 | 36.718 |
| FWD | 29.218 | 45.017 |

XGBoost wins for DEF (MAE 20.507 vs RF 23.833) in the comparison experiment but this was not the deployed model.

---

## Section 6: Model Comparison Summary Table

All metrics on the **2023-24 holdout test set** from `visuals/model_metrics_by_position.csv` and `visuals/advanced_model_metrics_by_position.csv`:

| Position | Experiment | Algorithm | CV MAE | Test MAE | Test RMSE | Sample Weight | Notes |
|----------|-----------|-----------|--------|----------|-----------|---------------|-------|
| GK | Comparison | Ridge | 23.652 | 16.835 | 27.419 | No | |
| GK | Comparison | RandomForest | 19.631 | 14.617 | 28.075 | No | **Best GK overall** |
| GK | Comparison | XGBoost | 19.513 | 15.448 | 28.939 | No | |
| GK | Comparison | LightGBM | 19.422 | 15.199 | 29.991 | No | |
| GK | Advanced | Ridge | 23.879 | 17.000 | 27.466 | No | |
| GK | Advanced | RandomForest | 19.164 | 15.168 | 28.776 | Yes | |
| GK | Advanced | XGBoost | 20.811 | 27.927 | 35.797 | Yes | Overfitting |
| GK | Advanced | LightGBM | 21.722 | 17.696 | 31.410 | Yes | |
| DEF | Comparison | Ridge | 25.194 | 21.047 | 25.908 | No | |
| DEF | Comparison | RandomForest | 24.838 | 23.833 | 28.777 | No | |
| DEF | Comparison | XGBoost | 24.759 | 20.507 | 26.514 | No | **Best DEF overall** |
| DEF | Comparison | LightGBM | 25.889 | 22.023 | 27.736 | No | |
| DEF | Advanced | Ridge | 25.316 | 21.004 | 26.030 | No | |
| DEF | Advanced | RandomForest | 25.102 | 22.260 | 27.875 | Yes | |
| DEF | Advanced | XGBoost | 28.097 | 26.331 | 31.273 | Yes | Sample weight hurts |
| DEF | Advanced | LightGBM | 28.734 | 27.649 | 33.507 | Yes | Sample weight hurts |
| MID | Comparison | Ridge | 26.223 | 26.570 | 36.592 | No | |
| MID | Comparison | RandomForest | 24.746 | 24.849 | 36.190 | No | **Best MID overall** |
| MID | Comparison | XGBoost | 25.170 | 26.300 | 36.718 | No | |
| MID | Comparison | LightGBM | 25.885 | 25.562 | 36.363 | No | |
| MID | Advanced | Ridge | 26.353 | 26.624 | 36.719 | No | |
| MID | Advanced | RandomForest | 25.089 | 27.805 | 36.865 | Yes | |
| MID | Advanced | XGBoost | 28.872 | 31.080 | 39.541 | Yes | Sample weight hurts |
| MID | Advanced | LightGBM | 29.188 | 30.431 | 38.455 | Yes | Sample weight hurts |
| FWD | Comparison | Ridge | 28.168 | 31.606 | 46.448 | No | |
| FWD | Comparison | RandomForest | 29.229 | 27.496 | 44.605 | No | **Best FWD overall** |
| FWD | Comparison | XGBoost | 29.016 | 29.218 | 45.017 | No | |
| FWD | Comparison | LightGBM | 30.144 | 27.664 | 43.189 | No | Close second |
| FWD | Advanced | Ridge | 28.168 | 31.622 | 46.523 | No | |
| FWD | Advanced | RandomForest | 29.607 | 28.345 | 44.973 | Yes | |
| FWD | Advanced | XGBoost | 31.439 | 29.415 | 43.874 | Yes | |
| FWD | Advanced | LightGBM | 33.498 | 32.753 | 48.668 | Yes | |

### Key Takeaways

- **RandomForest without sample weighting** is the best or tied-best for GK, MID, FWD
- **XGBoost without sample weighting** is the best for DEF
- Sample weighting consistently **hurt** performance in the advanced experiments for tree-based models
- Ridge is competitive for DEF but weaker elsewhere
- The deployed models (`position_model_*.joblib`) are basic RF with n_estimators=100 — likely slightly worse than the comparison RF results above

---

## Section 7: How to Retrain / Build a New Model

### Retrain Basic Models

```bash
source .venv/bin/activate
python -m training.train_position_models
```

This will:
1. Load seasons 2019-20 through 2022-23 from `data/historical_exports/`
2. Train 4 `RandomForestRegressor(n_estimators=100, random_state=42)` models
3. Save to `models/position_model_{1,2,3,4}.joblib`
4. Print MAE per position
5. Save a bar chart to `visuals/mae_by_position.png`

### Retrain Advanced Models

```bash
source .venv/bin/activate
python -m training.train_advanced_models
```

This will:
1. Auto-discover all seasons in `data/historical_exports/`
2. Use the latest as holdout (default 2023-24)
3. Train Ridge, RF, XGBoost, LightGBM per position with hyperparameter search
4. Save winning model per position to `models/advanced_model_{GK,DEF,MID,FWD}.joblib`
5. Save metrics to `visuals/advanced_model_metrics_by_position.csv`
6. Save predictions to `visuals/advanced_winner_predictions.csv`
7. Save feature importances to `visuals/advanced_winner_top_features.csv`
8. Save charts to `visuals/`

### Adding a New Model Type

To add a new algorithm to the comparison or advanced training:

1. **In `train_and_compare_models.py` or `train_advanced_models.py`:** Add an entry to the `model_registry()` function. Each entry is a tuple of `(Pipeline, param_grid, search_strategy)` (comparison) or `(Pipeline, param_grid, search_strategy, use_weights)` (advanced).

2. The Pipeline must have:
   - A `"preprocess"` step (imputation at minimum)
   - A `"model"` step (the sklearn-compatible estimator)

3. The param grid keys must be prefixed with `"model__"` (sklearn Pipeline convention).

### Deploying a New Model

1. Train and save the model as a `.joblib` file
2. Place it in `models/` with the naming convention:
   - `position_model_1.joblib` = GK
   - `position_model_2.joblib` = DEF
   - `position_model_3.joblib` = MID
   - `position_model_4.joblib` = FWD
3. The mapping is defined in `src/team_builder.py:32-37` as `MODEL_FILENAMES`
4. The model must support `.predict(X)` where X is a DataFrame with columns matching the output of `_build_feature_cols()` — i.e., `price_now` plus all numeric `1_years_past_*` columns
5. If the model is a sklearn Pipeline, `.predict()` works natively
6. If the model is a raw estimator, it must accept the same feature matrix (24 columns, NaN-filled with 0 by the caller)

### Model File Naming Convention

| File | Position | element_type |
|------|----------|-------------|
| `position_model_1.joblib` | GK | 1 |
| `position_model_2.joblib` | DEF | 2 |
| `position_model_3.joblib` | MID | 3 |
| `position_model_4.joblib` | FWD | 4 |

### Feature Contract

The model's `.predict(X)` will receive a DataFrame with these columns (order may vary):
- `price_now` (numeric, FPL tenths of £M)
- All `1_years_past_*` columns that are numeric in the training data (23 columns)
- NaN values are filled with 0 before calling predict (in `team_builder.py`)

If the model is a Pipeline with its own imputer, it will receive NaN values directly (the Pipeline handles imputation).

---

## Section 8: Gaps and Opportunities

### Data Available but Unused

1. **2-year lag features** (`2_years_past_*`) — 23 columns per player capturing performance from 2 seasons ago. Could help model career trajectories and distinguish one-season wonders from consistent performers.
2. **3-year lag features** (`3_years_past_*`) — 23 columns. Even deeper history.
3. **Season 2024-25 data** — Available in `data/historical_exports/historical_players_2024-25.csv` (807 rows) but not used for training in the basic/comparison models (only 2019-20 to 2022-23 train, 2023-24 test).
4. **`1_years_past_element_type`** — included in features but is just the position code. Redundant since models are already position-specific.

### Algorithms Not Tried

- **Neural networks** (MLP, simple feed-forward)
- **Stacking / blending ensembles** (combine RF + XGBoost + Ridge predictions)
- **CatBoost** (handles categoricals natively)
- **Bayesian hyperparameter optimization** (instead of random search with only 5 iterations)
- **Quantile regression** (predict confidence intervals, not just point estimates)

### Feature Engineering Opportunities

1. **Form trends:** Delta between 1-year and 2-year stats (improving vs declining players)
2. **Per-90 stats for more metrics:** Currently only total_points_per_90 and ict_index_per_90. Could compute goals_per_90, assists_per_90, bps_per_90, etc.
3. **Playing time stability:** Variance in minutes across gameweeks (from raw GW data if available)
4. **Price momentum:** `price_now - 1_years_past_now_cost` (market signal of expected improvement)
5. **Age** (not in current data, would need external source)
6. **Fixture difficulty** (FDR from FPL API — more relevant for gameweek prediction)
7. **Team strength proxy:** Average team total_points from prior season
8. **Binary "played last season" flag:** To help model distinguish non-playing from playing players (major error source)

### Baseline Performance (Beat These)

Any new model must beat the current best test MAE per position on the 2023-24 holdout:

| Position | Current Best MAE | Current Best Model | Current Best RMSE |
|----------|-----------------|-------------------|------------------|
| GK | 14.617 | RandomForest (comparison, no weights) | 28.075 |
| DEF | 20.507 | XGBoost (comparison, no weights) | 26.514 |
| MID | 24.849 | RandomForest (comparison, no weights) | 36.190 |
| FWD | 27.496 | RandomForest (comparison, no weights) | 44.605 |

**Weighted average baseline MAE** (by test set size): (14.617×100 + 20.507×285 + 24.849×374 + 27.496×113) / 872 = **22.38**

### Quick Wins

1. **Remove sample weighting** from the advanced experiment — it consistently hurt performance
2. **Use XGBoost for DEF** — already proven to beat RF by 3.3 MAE points
3. **Increase RandomizedSearchCV iterations** from 5 to 20+ — current search barely explores the space
4. **Add a "minutes > 0 last season" binary feature** — directly addresses the biggest error source
5. **Train on more seasons** (add 2023-24 to training, test on 2024-25) — more data usually helps
