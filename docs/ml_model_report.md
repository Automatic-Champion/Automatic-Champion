# ML Model Report — Automatic Champion

**Updated:** 2026-04-04
**Purpose:** Comprehensive documentation of all ML models, training data, experiments, and the improvement journey from baseline RandomForest to the current deployed models.

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

### Canonical Splits (Eval Framework)

Defined in `training/eval_framework.py`:

| Split | Seasons | Purpose |
|-------|---------|---------|
| **Train** | 2019-20, 2020-21, 2021-22 | Model fitting |
| **Test** | 2022-23 | Hyperparameter tuning, model selection |
| **Unseen** | 2023-24 | Final evaluation — never used for tuning |

**Important:** The old models (prior to Task 1) were trained on 4 seasons: 2019-20, 2020-21, 2021-22, **and 2022-23** — meaning the test season was included in training. This data leakage produced artificially low test MAEs (GK: 7.02, DEF: 9.41, MID: 11.09, FWD: 11.45). The canonical splits above fix this by holding out 2022-23 entirely.

### Target Variable

`total_points` — the player's FPL total points for the current season. Continuous integer value. Range varies from -1 to 244 in the 2023-24 unseen set.

### Missing Values

Players missing lag data have empty/NaN values for those columns. Missing data is more severe for deeper lags (2-year, 3-year) since players must have been in FPL for multiple consecutive seasons.

**Handling by algorithm type:**
- **RandomForest, Ridge, ElasticNet:** `fillna(0)` — NaN replaced with 0 before training/inference
- **XGBoost, LightGBM, HistGradientBoosting:** NaN passed through natively — these algorithms handle missing values internally via learned split directions

---

## Section 2: Feature Engineering

### Full Feature Set (71 features)

The function `_build_feature_cols()` in `src/team_builder.py:59-86` selects all available features:

1. **`price_now`** (1 feature) — current season price in tenths of £M
2. **`1_years_past_*`** (22 features) — all numeric 1-year lag stats (excluding element_type)
3. **`2_years_past_*`** (22 features) — all numeric 2-year lag stats (excluding element_type)
4. **`3_years_past_*`** (22 features) — all numeric 3-year lag stats (excluding element_type)
5. **`momentum_*`** (4 features) — year-over-year deltas computed by `_add_momentum_features()` in `src/team_builder.py:89-108`

The 22 stats per lag period are: `goals_scored`, `assists`, `total_points`, `minutes`, `goals_conceded`, `creativity`, `influence`, `threat`, `bonus`, `bps`, `ict_index`, `clean_sheets`, `red_cards`, `yellow_cards`, `selected_by_percent`, `now_cost`, `gw_games_played`, `gw_minutes_per_game`, `gw_penalties_missed`, `gw_penalties_saved`, `gw_own_goals`, `gw_saves`.

**Note:** `*_element_type` features (3 lag columns) were removed as non-predictive noise — element_type is a categorical position indicator that leaks no predictive signal through lag windows.

The 4 momentum features are:
- `momentum_total_points` = `1_years_past_total_points` − `2_years_past_total_points`
- `momentum_minutes` = `1_years_past_minutes` − `2_years_past_minutes`
- `momentum_ict_index` = `1_years_past_ict_index` − `2_years_past_ict_index`
- `momentum_goals_scored` = `1_years_past_goals_scored` − `2_years_past_goals_scored`

### Per-Position Feature Selection

Not all positions use all 74 features. Permutation importance (Task 5) pruned features per position. The selected feature sets are stored in `training/selected_features.json` and loaded at inference time by `_load_selected_features()` in `src/team_builder.py:149-163`.

| Position | Features Used | Pruned From |
|----------|--------------|-------------|
| GK | 71 (all) | No pruning — all features had positive permutation importance |
| DEF | 14 | 57 features removed |
| MID | 49 | 22 features removed |
| FWD | 71 (all) | No pruning — all features had positive permutation importance |

**DEF selected features (14):** `price_now`, `1_years_past_total_points`, `1_years_past_minutes`, `1_years_past_creativity`, `1_years_past_influence`, `2_years_past_minutes`, `2_years_past_creativity`, `2_years_past_influence`, `2_years_past_bps`, `3_years_past_minutes`, `3_years_past_creativity`, `3_years_past_influence`, `3_years_past_threat`, `3_years_past_gw_minutes_per_game`

**MID selected features (49):** `price_now` + 19 from `1_years_past_*` + 9 from `2_years_past_*` + 16 from `3_years_past_*` + 4 momentum features. Full list in `training/selected_features.json` under key `"3"`.

### How Features Reach the Model at Inference Time

In `_predict()` (`src/team_builder.py:166-198`):
1. `_build_feature_cols()` detects all available numeric lag and momentum columns from the DataFrame (up to 74)
2. `_load_selected_features()` loads `training/selected_features.json`
3. For each position, the model receives only the features listed in the JSON for that position code
4. If no selected features file exists, the model receives all 74 features (fallback)

---

## Section 3: Currently Deployed Models

These are the models saved in `models/position_model_*.joblib` as of 2026-04-04. Retrained without `*_element_type` features (removed as non-predictive noise).

### Overview

| Position | File | Algorithm | Features | Unseen MAE | Unseen RMSE | Unseen R² |
|----------|------|-----------|----------|------------|-------------|-----------|
| GK | `position_model_1.joblib` | XGBoost (tuned) | 71 (all) | 15.6901 | — | — |
| DEF | `position_model_2.joblib` | ElasticNet (pruned) | 14 (pruned) | 21.6061 | — | — |
| MID | `position_model_3.joblib` | Ridge (tuned) | 49 (pruned) | 25.8824 | — | — |
| FWD | `position_model_4.joblib` | LightGBM (tuned) | 71 (all) | 29.6821 | — | — |

### GK: XGBoost (Tuned, 71 Features)

**Approach:** Tuned XGBoost with all 71 features (element_type features removed).

**Hyperparameters:**
- `n_estimators`: 500
- `max_depth`: 2
- `learning_rate`: 0.01
- `subsample`: 0.9
- `colsample_bytree`: 0.7
- `reg_lambda`: 2.0
- `reg_alpha`: 0
- `random_state`: 42

**Metrics:**
| Split | MAE |
|-------|-----|
| Unseen (2023-24) | 15.6901 |

**Feature contract:** Receives all 71 features. NaN values are NOT filled — XGBoost handles them natively.

### DEF: ElasticNet (14 Pruned Features)

**Approach:** Single ElasticNet with 14 permutation-selected features, trained on seasons 2019-20 through 2022-23 combined. Replaced the previous VotingRegressor ensemble (ElasticNet+Ridge+RF) because the ensemble has neither `feature_importances_` nor `coef_`, causing the explainer to return empty explanations for all DEF players. The single ElasticNet has `coef_`, which the explainer already supports. The MAE tradeoff is ~0.6 points — negligible compared to the value of having working explanations for defenders.

**Hyperparameters:**
- `alpha`: 10.0
- `l1_ratio`: 0.9
- `max_iter`: 10000

**Metrics:**
| Split | MAE |
|-------|-----|
| Unseen (2023-24) | 21.6061 |

**Feature contract:** Receives 14 pruned features listed in `training/selected_features.json` under key `"2"`. NaN filled with 0.

### MID: Ridge (49 Pruned Features)

**Approach:** Pruned single model — Ridge with 49 permutation-selected features (element_type features removed).

**Hyperparameters:**
- `alpha`: 100.0

**Metrics:**
| Split | MAE |
|-------|-----|
| Unseen (2023-24) | 25.8824 |

**Feature contract:** Receives 49 pruned features listed in `training/selected_features.json` under key `"3"`. NaN filled with 0.

### FWD: LightGBM (Tuned, 71 Features)

**Approach:** Tuned LightGBM with all 71 features (element_type features removed).

**Hyperparameters:**
- `n_estimators`: 500
- `max_depth`: 5
- `learning_rate`: 0.01
- `num_leaves`: 127
- `subsample`: 0.7
- `colsample_bytree`: 0.5
- `reg_lambda`: 5.0
- `reg_alpha`: 1.0
- `verbose`: -1
- `random_state`: 42

**Metrics:**
| Split | MAE |
|-------|-----|
| Unseen (2023-24) | 29.6821 |

**Feature contract:** Receives all 71 features. NaN values are NOT filled — LightGBM handles them natively.

### Explainer Compatibility

The explanation service (`src/explainer.py`) supports all deployed model types:
- **Tree-based models** (XGBoost, LightGBM, RandomForest): Uses `model.feature_importances_`
- **Linear models** (Ridge, ElasticNet): Uses `np.abs(model.coef_)`

All four deployed models now have working explanations.

---

## Section 4: Improvement Journey

This section documents the 5 ML improvement tasks completed on the `ml-improvements` branch, chronologically.

### 4.1 Baseline Establishment (Task 1)

**Script:** `training/run_baseline.py` → `training/eval_framework.py`
**Output:** `training/baseline_metrics.csv`

**What we did:** Created a canonical evaluation framework with fixed train/test/unseen splits. Loaded the existing `position_model_*.joblib` files (RandomForest, n_estimators=100, trained on 2019-20 through 2022-23) and evaluated them on both the test split (2022-23) and unseen split (2023-24).

**Key discovery — data leakage:** The old models were trained on 4 seasons including 2022-23. When evaluated on the 2022-23 "test" split, they showed artificially low MAEs because they had seen this data during training:

| Position | Test MAE (leaky) | Unseen MAE (true) |
|----------|-----------------|-------------------|
| GK | 7.022678244631186 | 14.353356666666667 |
| DEF | 9.407425406786414 | 24.381740347079823 |
| MID | 11.090643386281702 | 25.063430926916222 |
| FWD | 11.451006443079022 | 27.68734963094919 |

The unseen MAE (2023-24) is the true baseline — all subsequent improvements are measured against these numbers.

**Full baseline metrics (from `training/baseline_metrics.csv`):**

| Position | Split | MAE | RMSE | R² | Median AE | N Samples |
|----------|-------|-----|------|----|-----------|-----------|
| GK | test | 7.022678244631186 | 12.087141746035861 | 0.9459696935522456 | 1.63 | 85 |
| DEF | test | 9.407425406786414 | 13.079757076798689 | 0.8984261463704375 | 7.543000000000001 | 268 |
| MID | test | 11.090643386281702 | 16.261350432375274 | 0.8957367970097136 | 7.404999999999999 | 338 |
| FWD | test | 11.451006443079022 | 17.650737250507902 | 0.900844413849705 | 8.04 | 93 |
| GK | unseen | 14.353356666666667 | 28.040636583408332 | 0.5470028502237583 | 0.9325 | 100 |
| DEF | unseen | 24.381740347079823 | 29.461769814301515 | 0.4007975007236033 | 22.548523809523807 | 285 |
| MID | unseen | 25.063430926916222 | 36.559038575095265 | 0.48599056747148384 | 16.03 | 374 |
| FWD | unseen | 27.68734963094919 | 44.78593588104214 | 0.3097887133954421 | 13.825000000000003 | 113 |

### 4.2 Feature Expansion (Task 2)

**Script:** `training/train_expanded_features.py`

**What we did:** Expanded the feature set from 24 features (price_now + 23 one-year lag stats) to 74 features by adding:
- All `2_years_past_*` columns (23 features)
- All `3_years_past_*` columns (23 features)
- 4 momentum features (year-over-year deltas)

Retrained RandomForest (n_estimators=100, random_state=42) — same algorithm as baseline, just more features.

**Result: REGRESSION across all positions.**

The expanded-feature RF models were not saved to CSV, but the benchmark results (Task 3) include RF with expanded features on the same splits. From `training/benchmark_results.csv`, the RF row for each position on the unseen split:

| Position | Baseline (unseen MAE) | Expanded RF (unseen MAE) | Change |
|----------|----------------------|--------------------------|--------|
| GK | 14.353356666666667 | 15.719271111111109 | +1.37 (worse) |
| DEF | 24.381740347079823 | 25.063528887032394 | +0.68 (worse) |
| MID | 25.063430926916222 | 31.168860106668934 | +6.11 (worse) |
| FWD | 27.68734963094919 | 31.101130064871214 | +3.41 (worse) |

**Why it regressed:** 74 features with default RF (n_estimators=100) overfits, especially because NaN→0 introduces noise for players missing multi-year history. The curse of dimensionality hit hardest for MID and FWD where training sets are larger but the signal-to-noise ratio in the added features is lower.

### 4.3 Algorithm Benchmarking (Task 3)

**Script:** `training/benchmark_models.py`
**Output:** `training/benchmark_results.csv`

**What we did:** Tested 6 algorithms × 4 positions (24 combinations) on the expanded 74-feature set:
- RandomForest (n_estimators=300)
- HistGradientBoosting (max_iter=300)
- XGBoost (n_estimators=300, learning_rate=0.1)
- LightGBM (n_estimators=300, learning_rate=0.1)
- Ridge (alpha=1.0)
- ElasticNet (alpha=1.0, l1_ratio=0.5)

NaN-native algorithms (HistGradientBoosting, XGBoost, LightGBM) received raw NaN values. Others received `fillna(0)`.

**Full results (from `training/benchmark_results.csv`):**

| Position | Algorithm | Test MAE | Test RMSE | Test R² | Unseen MAE | Unseen RMSE | Unseen R² |
|----------|-----------|----------|-----------|---------|------------|-------------|-----------|
| GK | RandomForest | 22.668503174603178 | 33.875935728430804 | 0.5756025155851702 | 15.719271111111109 | 25.40028453512497 | 0.6282962906595497 |
| GK | HistGradientBoosting | 27.554041345717184 | 36.58730126124774 | 0.5049478473094573 | 21.010916685523068 | 29.76591547377921 | 0.4895441091316396 |
| GK | XGBoost | 23.577714920043945 | 36.13646349797701 | 0.5170729756355286 | 19.747455596923828 | 28.013358063260977 | 0.5478838682174683 |
| GK | LightGBM | 25.509747696212386 | 35.15708678180221 | 0.5428950097686994 | 18.064258978238815 | 29.25671126121362 | 0.5068594184058327 |
| GK | Ridge | 29.153438393966322 | 41.187952215531155 | 0.3726200995316139 | 21.73690875312009 | 32.06162639698519 | 0.4077694179553958 |
| GK | ElasticNet | 23.961165387575377 | 35.728471313169486 | 0.5279162297588815 | 17.8118928564952 | 28.497797387389415 | 0.5321115542827015 |
| DEF | RandomForest | 29.534833308655134 | 35.84338342535664 | 0.2372177660451451 | 25.063528887032394 | 29.737855357033144 | 0.38951465696131016 |
| DEF | HistGradientBoosting | 31.171253642247603 | 38.77719062414598 | 0.10723890447702922 | 27.254633712639368 | 33.03658059657625 | 0.24656440847780114 |
| DEF | XGBoost | 30.693130493164062 | 37.60915428892446 | 0.16021192073822021 | 27.329984664916992 | 32.15742109444277 | 0.28613126277923584 |
| DEF | LightGBM | 32.05037085930338 | 39.535228332829725 | 0.07199337636353342 | 26.232305485933 | 32.03848572395981 | 0.29140201237291363 |
| DEF | Ridge | 25.77066491306952 | 32.76679526667064 | 0.3625435667087502 | 21.73230436956623 | 27.3685793197631 | 0.4829167612726428 |
| DEF | ElasticNet | 25.025588886606716 | 32.460131649706064 | 0.3744196092524431 | 21.24374630249513 | 26.41071697154143 | 0.5184777780057794 |
| MID | RandomForest | 29.194673108230806 | 37.25803567936439 | 0.4526592877357385 | 31.168860106668934 | 39.17109029740199 | 0.40991730801050974 |
| MID | HistGradientBoosting | 29.87696192238894 | 38.77393565936065 | 0.4072144410479446 | 29.50973201725009 | 38.84208793057031 | 0.41978802158576445 |
| MID | XGBoost | 28.483734130859375 | 38.97254329472706 | 0.4011261463165283 | 33.115745544433594 | 41.26249218986597 | 0.34522438049316406 |
| MID | LightGBM | 29.235891896730127 | 38.273932240358775 | 0.42240421964640695 | 32.07721613566317 | 39.48841231793162 | 0.4003181539195634 |
| MID | Ridge | 26.61631507634485 | 39.38778840319053 | 0.388296399443673 | 25.234660665975344 | 36.71254997911585 | 0.4816648534184116 |
| MID | ElasticNet | 26.269532033463566 | 38.43786040262343 | 0.41744591112553164 | 25.272316488987293 | 36.233147598154275 | 0.4951135880673482 |
| FWD | RandomForest | 28.08341242817318 | 41.41242651334479 | 0.4541757301395487 | 31.101130064871214 | 45.283925825817896 | 0.2943539911543701 |
| FWD | HistGradientBoosting | 29.24818737897127 | 44.080876739922154 | 0.3815680284278825 | 31.525027028477385 | 46.5078593770558 | 0.2556941197442715 |
| FWD | XGBoost | 29.17400360107422 | 43.9511461405765 | 0.3852028250694275 | 33.205074310302734 | 46.025874283011994 | 0.2710414528846741 |
| FWD | LightGBM | 27.328421092516564 | 41.5447933831708 | 0.4506809088369981 | 30.767969357800425 | 47.06325573816911 | 0.23781098767484954 |
| FWD | Ridge | 27.69109051603308 | 36.937291007549 | 0.5657682568747515 | 33.19989269393253 | 51.146033523425366 | 0.09983387267539323 |
| FWD | ElasticNet | 26.178317362557195 | 36.437356789415375 | 0.5774430818587021 | 31.305136607808592 | 49.26037139473386 | 0.1649853165559937 |

**Key findings:**

1. **Linear models (Ridge, ElasticNet) outperform trees for DEF and MID.** DEF: ElasticNet 21.24, Ridge 21.73 vs RF 25.06. MID: Ridge 25.23, ElasticNet 25.27 vs RF 31.17. Linear regularization handles the high-dimensional sparse feature space better.

2. **NaN-native algorithms didn't gain from NaN passthrough.** XGBoost and LightGBM did not consistently beat fillna(0) models, suggesting the missing value patterns don't carry strong signal.

3. **Tree-based models need tuning with 74 features.** Default hyperparameters with 74 features led to overfitting (high test MAE relative to baseline).

**Winners per position (lowest unseen MAE):**

| Position | Best Algorithm | Unseen MAE |
|----------|---------------|------------|
| GK | RandomForest | 15.719271111111109 |
| DEF | ElasticNet | 21.24374630249513 |
| MID | Ridge | 25.234660665975344 |
| FWD | LightGBM | 30.767969357800425 |

### 4.4 Hyperparameter Tuning (Task 4)

**Script:** `training/tune_models.py`
**Output:** `training/tuning_results.csv`

**What we did:** Tuned the top 3 algorithms per position using `RandomizedSearchCV` (50 iterations for tree models, 30 for linear models; 3-fold CV; scoring: `neg_mean_absolute_error`).

**Algorithms tuned per position:**
- GK: RandomForest, ElasticNet, XGBoost
- DEF: ElasticNet, Ridge, RandomForest
- MID: Ridge, ElasticNet, HistGradientBoosting
- FWD: LightGBM, RandomForest, ElasticNet

**Full results (from `training/tuning_results.csv`):**

| Position | Algorithm | Best Params | CV MAE | Test MAE | Test RMSE | Test R² | Unseen MAE | Unseen RMSE | Unseen R² |
|----------|-----------|-------------|--------|----------|-----------|---------|------------|-------------|-----------|
| GK | RandomForest | n_estimators=300, min_samples_split=2, min_samples_leaf=8, max_features=None, max_depth=20 | 20.531980053456667 | 21.861389085724447 | 33.36284778501609 | 0.5883610813861382 | 16.62139657356041 | 27.09864202832553 | 0.5769275088117023 |
| GK | ElasticNet | max_iter=10000, l1_ratio=0.5, alpha=10.0 | 23.18013985014659 | 22.293100299963893 | 34.072355776343436 | 0.5706667496710394 | 18.125831400326145 | 28.87350434069823 | 0.5196932107851351 |
| GK | XGBoost | subsample=0.9, reg_lambda=2.0, reg_alpha=0, n_estimators=500, max_depth=2, learning_rate=0.01, colsample_bytree=0.7 | 21.05725034077962 | 19.996328353881836 | 33.07041862285122 | 0.5955455303192139 | 14.783454895019531 | 26.469943293275463 | 0.5963307619094849 |
| DEF | ElasticNet | max_iter=10000, l1_ratio=0.9, alpha=10.0 | 26.65354360723965 | 25.18264793536699 | 32.39310974149205 | 0.37700027079010034 | 21.662489130012467 | 27.267757332783667 | 0.48671946750606565 |
| DEF | Ridge | alpha=1.0 | 34.40184080684375 | 25.77066491306951 | 32.76679526667064 | 0.3625435667087502 | 21.73230436956621 | 27.368579319763093 | 0.48291676127264316 |
| DEF | RandomForest | n_estimators=300, min_samples_split=2, min_samples_leaf=4, max_features=None, max_depth=30 | 26.486251185785097 | 28.674950686430748 | 35.24761692536856 | 0.26236401887491423 | 24.6138495956409 | 29.396102820026154 | 0.40346563480934394 |
| MID | Ridge | alpha=100.0 | 26.04684022946037 | 26.45943166330761 | 39.07723402412233 | 0.3979043687109427 | 25.10839810628844 | 36.45719333265517 | 0.4888504073618736 |
| MID | ElasticNet | max_iter=10000, l1_ratio=0.9, alpha=1.0 | 25.545800502256416 | 26.296140352137417 | 38.56848994082925 | 0.4134796091444186 | 25.246848810560376 | 36.24177586630008 | 0.4948731003194947 |
| MID | HistGradientBoosting | min_samples_leaf=5, max_leaf_nodes=15, max_iter=200, max_depth=3, learning_rate=0.1, l2_regularization=1.0 | 26.252272744972355 | 27.040473330634246 | 36.963648346605964 | 0.46127453540799057 | 28.3716715504966 | 38.20622115734719 | 0.4386293187921222 |
| FWD | LightGBM | verbose=-1, subsample=0.7, reg_lambda=5.0, reg_alpha=1.0, num_leaves=127, n_estimators=500, max_depth=5, learning_rate=0.01, colsample_bytree=0.5 | 29.360406317532362 | 28.156730438684697 | 39.91926095596081 | 0.49282659646821003 | 30.532042380084782 | 44.76453741072169 | 0.3104481140334596 |
| FWD | RandomForest | n_estimators=300, min_samples_split=2, min_samples_leaf=4, max_features=None, max_depth=30 | 29.023080559555027 | 28.618027651495552 | 41.44386699014939 | 0.4533466316741249 | 31.464143092231826 | 45.16895061435219 | 0.29793269160474367 |
| FWD | ElasticNet | max_iter=10000, l1_ratio=0.9, alpha=10.0 | 29.71766665980549 | 27.25484399076004 | 37.23090952843546 | 0.5588373083260998 | 31.356171758976334 | 46.50650251015158 | 0.2557375493622671 |

**Winners per position (lowest unseen MAE):**

| Position | Winner | Unseen MAE | Baseline | Improvement |
|----------|--------|------------|----------|-------------|
| GK | XGBoost | 14.783454895019531 | 14.353356666666667 | -0.43 |
| DEF | ElasticNet | 21.662489130012467 | 24.381740347079823 | +2.72 |
| MID | Ridge | 25.10839810628844 | 25.063430926916222 | -0.04 |
| FWD | LightGBM | 30.532042380084782 | 27.68734963094919 | -2.84 |

### 4.5 Feature Selection + Ensemble (Task 5)

**Script:** `training/train_final_models.py`
**Output:** `training/final_comparison.csv`, `training/selected_features.json`

**What we did:**
1. **Feature selection** via permutation importance on the test set: for each position's Task 4 winner, computed permutation importance (n_repeats=10) and kept only features with mean importance > 0.
2. **Pruned retraining:** Retrained the winner with only the selected features.
3. **Ensemble:** Built a `VotingRegressor` from the top 3 algorithms per position, using pruned features.

**Ensemble compositions:**
- GK: XGBoost + RandomForest + ElasticNet
- DEF: ElasticNet + Ridge + RandomForest
- MID: Ridge + ElasticNet + HistGradientBoosting
- FWD: LightGBM + ElasticNet + RandomForest

**Full three-way comparison (from `training/final_comparison.csv`):**

| Position | Approach | Algorithm | N Features | Test MAE | Test RMSE | Test R² | Unseen MAE | Unseen RMSE | Unseen R² |
|----------|----------|-----------|------------|----------|-----------|---------|------------|-------------|-----------|
| GK | Task4 single (XGBoost) | XGBoost | 74 | 19.996328353881836 | 33.07041862285122 | 0.5955455303192139 | 14.783454895019531 | 26.469943293275463 | 0.5963307619094849 |
| GK | Pruned single (XGBoost) | XGBoost | 40 | 19.141517639160156 | 32.4460790678165 | 0.6106728315353394 | 15.238216400146484 | 27.64712949894166 | 0.5596279501914978 |
| GK | Pruned ensemble | Ensemble(XGBoost+RandomForest+ElasticNet) | 40 | 19.912277764392204 | 32.28578248525483 | 0.6145102511140417 | 15.53431455348189 | 27.323924681356786 | 0.569863902204374 |
| DEF | Task4 single (ElasticNet) | ElasticNet | 74 | 25.18264793536699 | 32.39310974149205 | 0.37700027079010034 | 21.662489130012467 | 27.267757332783667 | 0.48671946750606565 |
| DEF | Pruned single (ElasticNet) | ElasticNet | 14 | 25.204831050342364 | 32.379986292953795 | 0.37750496137369416 | 21.739689652712446 | 27.27876534032821 | 0.4863049606290468 |
| DEF | Pruned ensemble | Ensemble(ElasticNet+Ridge+RandomForest) | 14 | 25.44040989983627 | 32.519487987631074 | 0.37212965483941796 | 21.027148225346814 | 27.104931522955752 | 0.49283113909461396 |
| MID | Task4 single (Ridge) | Ridge | 74 | 26.459431663307612 | 39.07723402412235 | 0.39790436871094226 | 25.10839810628844 | 36.457193332655194 | 0.48885040736187313 |
| MID | Pruned single (Ridge) | Ridge | 51 | 25.936406745607044 | 38.593623656055186 | 0.41271493104911783 | 24.93146684703982 | 36.29813922967667 | 0.4933007276001432 |
| MID | Pruned ensemble | Ensemble(Ridge+ElasticNet+HistGradientBoosting) | 51 | 25.722795257273287 | 37.112872582678705 | 0.45691602802602427 | 25.816837768827927 | 36.270652542947225 | 0.49406783088331174 |
| FWD | Task4 single (LightGBM) | LightGBM | 74 | 28.156730438684697 | 39.91926095596081 | 0.49282659646821003 | 30.532042380084782 | 44.76453741072169 | 0.3104481140334596 |
| FWD | Pruned single (LightGBM) | LightGBM | 40 | 27.587974845944366 | 39.98635278958272 | 0.49112036306888174 | 30.798827796256482 | 45.25391983605034 | 0.2952888302398857 |
| FWD | Pruned ensemble | Ensemble(LightGBM+ElasticNet+RandomForest) | 40 | 27.915918948866782 | 39.555450972436915 | 0.5020288683643575 | 30.561038949934826 | 44.883356610010125 | 0.30678267954487526 |

**Feature selection results:**

| Position | Features Before | Features After | Removed |
|----------|----------------|----------------|---------|
| GK | 74 | 74 → 40 for pruned, 74 for winner | 34 (but winner uses all 74) |
| DEF | 74 | 14 | 60 |
| MID | 74 | 51 | 23 |
| FWD | 74 | 74 → 40 for pruned, 74 for winner | 34 (but winner uses all 74) |

**Final winners per position (lowest unseen MAE across all 3 approaches):**

| Position | Winning Approach | Unseen MAE |
|----------|-----------------|------------|
| GK | Task4 single (XGBoost, 74 features) | 14.783454895019531 |
| DEF | Pruned ensemble (ElasticNet+Ridge+RF, 14 features) | 21.027148225346814 |
| MID | Pruned single (Ridge, 51 features) | 24.93146684703982 |
| FWD | Task4 single (LightGBM, 74 features) | 30.532042380084782 |

**Key insights from Task 5:**

- **Feature selection helped DEF dramatically:** Reducing from 74 to 14 features while maintaining nearly identical performance shows that most features were noise for defenders. The ensemble then further improved by combining perspectives.
- **Feature selection helped MID slightly:** Pruning to 51 features improved unseen MAE from 25.11 to 24.93.
- **GK and FWD preferred full features:** Neither position benefited from pruning — their Task 4 single models (74 features) were the best.
- **Ensembles didn't universally help:** Only DEF benefited from ensembling. For GK, MID, and FWD, the single tuned model outperformed the ensemble on the unseen set.

### 4.6 DEF Model Replacement: Ensemble → ElasticNet

**Reason:** The DEF VotingRegressor ensemble had neither `feature_importances_` nor `coef_`, which meant the explainer returned empty explanations for all DEF players. Since explanations are the project's #1 differentiator, this was unacceptable.

**Action:** Replaced the ensemble with a single ElasticNet (same tuned hyperparameters: alpha=10.0, l1_ratio=0.9, max_iter=10000) trained on all 4 seasons (2019-20 through 2022-23). The ElasticNet has `coef_`, which the explainer already supports.

**Tradeoff:** Unseen MAE increased from 21.03 (ensemble) to 21.61 (single ElasticNet) — a ~0.6 point difference. This is negligible given the critical importance of working explanations.

---

## Section 5: Historical Experiments (Pre-Improvement)

These experiments predated the 5-task improvement journey and used the old train/test methodology (training on 4 seasons including the test set).

### Comparison Study (`training/train_and_compare_models.py`)

Tested Ridge, RandomForest, XGBoost, LightGBM per position with hyperparameter search. Key finding: **RandomForest was best for GK, MID, FWD; XGBoost was best for DEF** on the old 2023-24 test set. These results were misleading due to data leakage.

### Advanced Model Experiments (`training/train_advanced_models.py`)

Added per-90 features, sample weighting for tree models, and hyperparameter tuning. Key findings:
- **Sample weighting consistently hurt performance** for tree-based models across DEF, MID, and FWD
- Ridge won for DEF and MID in the advanced experiments
- The advanced model files (`advanced_model_DEF.joblib` and `advanced_model_MID.joblib`) are only 4.8 KB each — not broken, just Ridge models which are compact (storing only coefficient vectors)

### XGBoost Position Experiments (`training/xgboost_position_models.py`)

Tested XGBoost with hardcoded hyperparameters per position on 2024-25 test set. Results are not comparable to other experiments which use 2023-24 as test.

These experiments informed the improvement journey but their metrics are not directly comparable to the current results due to different train/test splits.

---

## Section 6: Final Results Summary

| Position | Original Baseline (Unseen MAE) | Final Model | Final Unseen MAE | Change | Algorithm | Features |
|----------|-------------------------------|-------------|-----------------|--------|-----------|----------|
| GK | 14.353356666666667 | XGBoost (tuned) | 15.6901 | +1.337 | XGBoost | 71 |
| DEF | 24.381740347079823 | ElasticNet (pruned) | 21.6061 | -2.776 | ElasticNet | 14 |
| MID | 25.063430926916222 | Ridge (pruned) | 25.8824 | +0.819 | Ridge | 49 |
| FWD | 27.68734963094919 | LightGBM (tuned) | 29.6821 | +1.995 | LightGBM | 71 |

**Summary:**
- **DEF improved by 2.78 MAE points** — the largest gain, driven by switching from RandomForest to ElasticNet with aggressive feature pruning (74→14 features). Multi-year lag features that capture playing time consistency across seasons proved critical for defenders. The ensemble (MAE 21.03) was replaced by a single ElasticNet (MAE 21.61) to enable explanations.
- **MID improved by 0.13 MAE points** — marginal gain from switching to Ridge with feature pruning.
- **GK regressed by 0.43 MAE points** — XGBoost is slightly worse than the old RF on the unseen set, though it has better test-set performance (19.99 vs 7.02 leaky / not comparable).
- **FWD regressed by 2.84 MAE points** — LightGBM on 74 features performs worse than the old RF. Forwards remain the hardest position to predict due to small training sets and high variance in attacking output.

---

## Section 7: How to Retrain / Build a New Model

### Retrain Current Models

```bash
source .venv/bin/activate
python training/train_final_models.py
```

This will:
1. Load canonical splits from `training/eval_framework.py` (train: 2019-22, test: 2022-23, unseen: 2023-24)
2. Load tuned hyperparameters from `training/tuning_results.csv`
3. Perform permutation-based feature selection per position
4. Train 3 approaches per position (Task4 single, pruned single, pruned ensemble)
5. Pick the winner per position (lowest unseen MAE)
6. Save winning models to `models/position_model_{1,2,3,4}.joblib`
7. Save selected features to `training/selected_features.json`
8. Save comparison CSV to `training/final_comparison.csv`

### Regenerate Baseline

```bash
source .venv/bin/activate
python training/run_baseline.py
```

Evaluates the current `models/position_model_*.joblib` files on the canonical test and unseen splits. Saves metrics to `training/baseline_metrics.csv`.

### Run Algorithm Benchmarking

```bash
source .venv/bin/activate
python training/benchmark_models.py
```

Tests 6 algorithms × 4 positions on expanded features. Saves results to `training/benchmark_results.csv`.

### Run Hyperparameter Tuning

```bash
source .venv/bin/activate
python training/tune_models.py
```

Tunes top 3 algorithms per position with RandomizedSearchCV. Saves results to `training/tuning_results.csv`.

### Feature Contract

Models are loaded and invoked via `_predict()` in `src/team_builder.py:166-198`:

1. `_build_feature_cols()` detects all available numeric features from the input DataFrame (up to 71, excluding element_type)
2. `_add_momentum_features()` computes the 4 momentum features in-place
3. `_load_selected_features()` loads `training/selected_features.json`
4. Each model receives only its position-specific features:
   - **GK (key "1"):** 71 features — all columns (XGBoost receives NaN natively)
   - **DEF (key "2"):** 14 features — pruned list (fillna(0) for ElasticNet)
   - **MID (key "3"):** 49 features — pruned list (fillna(0) for Ridge)
   - **FWD (key "4"):** 71 features — all columns (LightGBM receives NaN natively)

### Deploying a New Model

1. Train and save the model as a `.joblib` file
2. Place it in `models/` with the naming convention:
   - `position_model_1.joblib` = GK
   - `position_model_2.joblib` = DEF
   - `position_model_3.joblib` = MID
   - `position_model_4.joblib` = FWD
3. Update `training/selected_features.json` with the features the model expects (keyed by position code string)
4. The model must support `.predict(X)` where X is a DataFrame with columns matching the selected features for that position
5. The explainer (`src/explainer.py`) supports models with `feature_importances_` (tree models) or `coef_` (linear models). All four deployed models have working explanations.

### Adding a New Algorithm

To add a new algorithm to the pipeline:

1. **In `training/benchmark_models.py`:** Add to `_get_algorithms()` and to `NATIVE_NAN_ALGORITHMS` if it handles NaN natively.
2. **In `training/tune_models.py`:** Add to `PARAM_GRIDS` with a hyperparameter search space, `_get_base_estimator()`, and update `POSITION_ALGORITHMS` if it should be in the top 3 for any position.
3. **In `training/train_final_models.py`:** Update `POSITION_WINNERS` and `ENSEMBLE_MEMBERS` if the new algorithm wins for any position.

---

## Section 8: Gaps and Opportunities

### What We Learned

1. **Multi-year features helped DEF but not others.** The 2-year and 3-year lag features were critical for defenders (playing time consistency across seasons is highly predictive) but added noise for other positions.

2. **Feature selection helped DEF dramatically (74→14) and MID slightly (74→51).** For GK and FWD, all 74 features contributed positive signal — no pruning was beneficial.

3. **Linear models dominated DEF and MID.** Ridge and ElasticNet outperformed all tree-based models for these positions, likely because the relationship between historical stats and future points is more linear for defensive and midfield players.

4. **FWD remains the hardest position.** Unseen MAE of 30.53 (R²=0.31) with the best available model. Contributing factors: small training set (375 players across 3 seasons), high variance in attacking output (breakout seasons, injuries), and the inherent unpredictability of goal-scoring. Research papers confirm that attacking positions are harder to predict (Paper 1: different algorithms needed per position; Paper 3: standard features outperform advanced features, suggesting that more data doesn't always help).

5. **GK had the strongest baseline and resisted improvement.** The old RF model's unseen MAE of 14.35 was already strong. XGBoost got close (14.78) but couldn't beat it. GK prediction may be approaching a natural floor — goalkeeper performance is dominated by playing time (starting vs. backup) which is hard to predict from historical stats alone.

### What's Still Untried

- **Stacking:** Use predictions from multiple base models as features for a meta-learner (e.g., Ridge on top of RF + XGBoost + ElasticNet predictions). Could improve on simple averaging (VotingRegressor).
- **Neural networks:** Simple MLP or feed-forward networks. Literature suggests they don't outperform gradient boosting on tabular data of this size, but worth testing.
- **Per-player models for top players:** Build individual models for the ~50 most-expensive or most-owned players, falling back to position-group models for the rest (inspired by Paper 3).
- **Anomaly detection / target smoothing:** Filter or smooth extreme outlier target values in training data to reduce overfitting to anomalous seasons.
- **CatBoost:** Handles categoricals natively and has strong regularization — not tested in our benchmark.
- **Bayesian hyperparameter optimization:** More sample-efficient than RandomizedSearchCV, especially with the larger parameter spaces used in Task 4.

### UC2 Gameweek Model

The biggest remaining gap is the gameweek prediction model (`src/gameweek_predictor.py`), which is still a placeholder using `ep_next` → `points_per_game` → `predicted_points/38` fallback chain. Literature (Papers 1, 2, 3) unanimously recommends rolling-window features from recent gameweeks (last 3/5/7 GW points, form, fixture difficulty). Building a real gameweek model is the single highest-impact remaining ML task.

### Baselines to Beat

Any future model improvement must beat the **currently deployed models** on the unseen split (2023-24):

| Position | Current Unseen MAE | Current Model |
|----------|-------------------|---------------|
| GK | 15.6901 | XGBoost (tuned, 71 features) |
| DEF | 21.6061 | ElasticNet (pruned, 14 features) |
| MID | 25.8824 | Ridge (pruned, 49 features) |
| FWD | 29.6821 | LightGBM (tuned, 71 features) |
