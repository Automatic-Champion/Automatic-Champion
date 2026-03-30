# Literature Review Report — Automatic Champion (FPL Optimizer)

**Prepared for:** ML Partner  
**Date:** March 2026  
**Purpose:** Guide model improvement work based on findings from four research papers

---

## Paper 1: Optimising Daily Fantasy Sports Teams with Artificial Intelligence

**Authors:** Beal, Norman & Ramchurn (University of Southampton, 2020)  
**Published in:** International Journal of Computer Science in Sport, Vol. 19

### 1. Summary

This paper tackles the prediction + optimization pipeline for NFL Daily Fantasy Sports (DFS) on FanDuel. The authors use time-series features (prior N game-weeks of fantasy scores) to predict per-gameweek player points, testing Linear Regression, RBF, LSTM, and Random Forest. They then formulate team selection as a 0-1 knapsack / Mixed-Integer Program (MIP) solved with CPLEX. Tested on 4 NFL seasons (2014–2017), their combined system achieved a profit in 81.3% of game-weeks, with prediction RMSE 15.9% lower than the best prior benchmark. A key finding is that optimization quality matters more than prediction quality — using MIP with even mediocre predictions beat using good predictions with human-style team selection.

### 2. Relevance to Our Project

**Rating: HIGH**

This is the closest architectural match to Automatic Champion: predict player points → feed into ILP → select optimal squad. The paper validates that our predict-then-optimize pipeline is the right approach. The finding that optimization matters more than prediction accuracy is directly relevant — it confirms that our OR-Tools ILP is a critical strength, and that we should invest in improving it (or at least not break it).

### 3. Concrete Takeaways

- **Time-series features for gameweek prediction (UC2):** The paper uses the last N game-weeks of fantasy scores as features (testing 2–6 weeks). For our UC2 gameweek predictor (currently a placeholder), we could build rolling-window features from the FPL API's gameweek-by-gameweek history: `last_3_gw_points`, `last_5_gw_points`, etc. These are computable from `player.gw_history` via the live API.

- **Position-specific model selection:** The paper finds that different algorithms work best for different positions (e.g., Linear Regression for QB/WR/TE/DEF, Random Forest for RB/K). This mirrors our current 4-model approach (GK/DEF/MID/FWD). We should benchmark additional algorithms per position rather than using RF for all.

- **MIP vs. brute-force optimization:** Their MIP solved in 18.5ms vs. 396ms for brute force, and found equal or better solutions. Our OR-Tools ILP is already the right approach — this paper validates it. Average runtime of 18.5ms confirms that 30-second timeout is extremely generous.

- **Optimization > Prediction insight:** The "AI Optimisation" model (MIP + human-quality predictions) outperformed the "AI Predictions" model (ML predictions + human-style team picking). Score difference: 18.4 points between "All AI" and "No AI" models, with optimization contributing more than prediction. This means improving our ILP formulation (e.g., adding transfer constraints, multi-week planning) could be as impactful as improving MAE.

### 4. Implementation Difficulty

- Rolling GW features for UC2: **EASY** (< 1 day) — simple pandas rolling means on gameweek data
- Position-specific algorithm benchmarking: **MEDIUM** (1–3 days) — need to test LR, GBM, etc. per position with cross-validation
- MIP validation is already done (we use OR-Tools): **N/A**

### 5. Expected Impact

- Rolling GW features: Would fill the UC2 gap entirely. No MAE comparison possible yet (no UC2 baseline), but this is the foundation for any gameweek model.
- Position-specific algorithms: Could reduce season MAE by 1–3 points per position if a better algorithm exists for some positions (e.g., GBM for FWD where variance is highest).
- The optimization insight doesn't change MAE but reinforces that our ILP is a major differentiator — worth emphasizing in the defense presentation.

---

## Paper 2: Optimizing Fantasy Sports Team Selection with Deep Reinforcement Learning

**Authors:** Bhattacharjee, Marathe, Patil & Kapoor (Dream11, 2024)  
**Published in:** CODS-COMAD Dec '24

### 1. Summary

This paper from Dream11 (India's largest fantasy sports platform, 200M+ users) applies Deep Reinforcement Learning — specifically DQN and PPO — to fantasy cricket team selection. Instead of predict-then-optimize, they frame team creation as a sequential decision-making problem: an RL agent starts with a random 11-player team and iteratively swaps players to maximize fantasy points. They use 90-day rolling averages of 10 performance features per player. The PPO agent achieved the 67th percentile among real users on average across 4 cross-validation folds, outperforming RF classifiers (0.57 avg), SVM (0.55 avg), and previous-performance baselines (0.54 avg).

### 2. Relevance to Our Project

**Rating: LOW**

The RL approach replaces both prediction and optimization with a single learned policy. While intellectually interesting, this is fundamentally different from our architecture and has several problems for our use case:
- FPL has a 15-player squad with specific constraints (budget, max 3/club, position counts) — RL would need to learn these hard constraints implicitly, whereas our ILP guarantees feasibility.
- The paper's results (67th percentile) are modest — they don't compare to ILP-based approaches.
- RL requires massive training (2M timesteps, multiple GPUs on Databricks) — far beyond B.Sc. scope.
- The paper targets cricket, and the action space (swapping players in a 22-player pool) is much smaller than FPL's ~700 player pool.

### 3. Concrete Takeaways

- **90-day rolling feature window:** The idea of using a fixed rolling window of recent performance (rather than full-season stats) is applicable to our UC2 gameweek prediction. While we already have `1_years_past_*` features, a shorter rolling window (e.g., last 5–10 gameweeks) would capture current form better.

- **Comparison baselines:** Their "Previous Performance" and "Player % selection" baselines are useful sanity checks. For our project, we could compare our ILP output against a naive "pick the highest ep_next players that fit constraints" baseline and against FPL's "most-selected" squad.

- **What NOT to do:** The RL approach is explicitly not recommended for our project — it's too complex, too slow to train, and doesn't guarantee constraint satisfaction.

### 4. Implementation Difficulty

- Rolling window features: **EASY** (< 1 day)
- RL-based team selection: **HARD** (> 3 days, realistically weeks) — not recommended

### 5. Expected Impact

- Rolling window features would improve UC2 gameweek prediction (form-capturing). No direct season MAE impact.
- The RL approach itself is not expected to beat our ILP for squad selection, and would sacrifice explainability — our #1 differentiator.

---

## Paper 3: An Innovative Method for Accurate NBA Player Performance Forecasting and Line-up Optimization in Daily Fantasy Sports

**Authors:** Papageorgiou, Sarlis & Tjortjis (International Hellenic University, 2024)  
**Published in:** International Journal of Data Science and Analytics

### 1. Summary

This paper builds individualized ML models for 203 NBA players to predict daily Fantasy Points, using 10 seasons of standard and advanced basketball metrics. They test 14 ML algorithms plus a Voting Meta-Model (ensemble of top-3 models per player) across 4 scenarios varying feature sets (standard vs. advanced) and time periods (last 3 seasons vs. all 10 seasons). Their best approach achieved validation MAPE of 28.30% and MAE of 6.98 for FP prediction. They then apply linear optimization (PuLP library) to select an optimal 8-player DFS lineup subject to salary cap and position constraints. Tested against 11,764 real users on DraftKings, their Daily Line-up Optimizer ranked in the top 18.4%, with profitable lineups reaching top 23.5%.

### 2. Relevance to Our Project

**Rating: HIGH**

This paper is the most methodologically rich for our ML work. The core approach — per-entity modeling with feature engineering, multiple algorithms, and linear optimization — maps directly onto our pipeline. Key parallels: they use historical stats + lag features → ML prediction → linear optimization with constraints (salary cap ≈ our budget, position requirements ≈ ours, max from same team ≈ our max 3/club).

### 3. Concrete Takeaways

- **Lag features (1, 3, 5, 7, 10-game lags):** They create 1-game-lag features for all stats plus 3, 5, 7, and 10-game-lag averages for the target variable (Fantasy Points). For our project: we have `1_years_past_*` features but don't use multi-season lags. We should add `2_years_past_*` and `3_years_past_*` features (already available in our data but unused). For UC2, create rolling averages over the last 3, 5, and 7 gameweeks from FPL API data.

- **Momentum features:** They compute the difference between the current game and the previous game for all features. For FPL, this would be `current_gw_points - previous_gw_points`, `current_form - previous_form`, etc. This captures whether a player is trending up or down.

- **Anomaly detection on target variable:** They apply statistical anomaly detection (using standard deviation with a −2 boundary) on Fantasy Points, creating a "smoothed FP" feature. For FPL, we could detect anomalous gameweeks (very low scores due to early substitution, red cards) and smooth them to avoid training on noisy outliers.

- **Per-player modeling (selective):** They build individual models for each player. This is computationally expensive for ~700 FPL players, but we could do it for the top ~50 most-owned or most-expensive players, falling back to position-group models for the rest.

- **Voting Meta-Model / Ensemble:** Their best-performing model was often a Voting Meta-Model that ensembles the top-3 models per player. For us: after training RF, we should also train GBM and Ridge/ElasticNet per position, then create a simple averaging ensemble. Scikit-learn's `VotingRegressor` does this in ~5 lines of code.

- **Standard features outperformed advanced features:** Their "core" dataset (standard stats only) outperformed the "complete" dataset (standard + advanced) by 1.7–1.9% MAPE in evaluation and 0.2–2.1% in validation. This suggests that adding too many features can hurt — we should be cautious about feature bloat and focus on feature selection.

- **Two-step evaluation (train/test + unseen validation):** They split data 70/20/10 (train/test/unseen) and report metrics on both test and unseen sets. We should adopt this to verify our models don't overfit — currently we only report test MAE.

- **Rest days feature:** They calculate days between games. In FPL, the equivalent is time between gameweeks or whether a player had a midweek fixture (fatigue). Available from the FPL fixtures API.

- **Opponent features:** They include the opponent team's recent performance stats. In FPL, this maps to Fixture Difficulty Rating (FDR) which we already have access to via the API but aren't using in the season model.

- **Linear optimization with PuLP:** They use PuLP for lineup optimization. We use OR-Tools, which is equivalent — no change needed. Their constraints (salary cap, position, multi-game diversity) mirror ours (budget, position counts, max 3/club).

### 4. Implementation Difficulty

- Add 2-year and 3-year lag features: **EASY** (< 1 day) — data already exists, just add columns to feature matrix
- Momentum features: **EASY** (< 1 day) — simple differencing
- Anomaly detection / target smoothing: **EASY** (< 1 day) — standard deviation filter
- VotingRegressor ensemble: **EASY** (< 1 day) — `sklearn.ensemble.VotingRegressor`
- Opponent features (FDR): **EASY** (< 1 day) — already in FPL API
- Two-step validation split: **EASY** (< 1 day) — modify train/test split
- Per-player modeling for top players: **MEDIUM** (2–3 days) — need infrastructure for many models
- Full feature selection pipeline: **MEDIUM** (1–2 days)

### 5. Expected Impact

- **Adding 2-year and 3-year lags:** Likely to reduce season MAE by 1–3 points, especially for DEF/MID/FWD where consistency across seasons is predictive. GK might benefit less (keeper changes are rarer).
- **VotingRegressor ensemble:** Expected 1–2 point MAE improvement based on the paper's finding that ensembles were the most frequently best-performing model.
- **Feature selection (removing noise):** Could improve MAE by 0.5–1 point by preventing overfitting.
- **Opponent features (FDR):** More impactful for UC2 (gameweek prediction) than UC1 (season prediction), but fixture difficulty over a season could help UC1 too.
- **Two-step validation:** Improves robustness assessment, not MAE directly — but critical for the defense presentation.

---

## Paper 4: InferDB: In-Database Machine Learning Inference Using Indexes

**Authors:** Salazar-Díaz, Glavic & Rabl (Hasso Plattner Institute / UIC, 2024)  
**Published in:** PVLDB, Vol. 17

### 1. Summary

InferDB replaces the standard ML inference pipeline (preprocessing + model prediction) with a lightweight index-based lookup. It discretizes input features using supervised binning (OptBinning), selects the most predictive discretized features via a greedy heuristic based on Information Value (IV), then builds a database index mapping discretized feature combinations to aggregated model predictions. At inference time, a test point is mapped to the embedding space and looked up in the index, replacing both preprocessing and model evaluation. Tested on 6 datasets (regression, binary/multi-label classification), InferDB achieves competitive accuracy while reducing inference latency by up to 2 orders of magnitude. Implemented in Postgres using standard B-tree/trie indexes.

### 2. Relevance to Our Project

**Rating: LOW**

InferDB solves an inference speed problem — making predictions faster by replacing the model with index lookups. This is relevant to production systems making millions of predictions per second. Our project makes predictions for ~700 players, which takes well under a second with scikit-learn. The latency optimization is irrelevant for our scale. However, there are two minor insights worth noting.

### 3. Concrete Takeaways

- **Supervised discretization / binning for feature engineering:** InferDB's use of OptBinning to discretize features while preserving predictive power is an interesting technique. For our project, we could bin continuous features like `price_now` or `minutes` into categories (e.g., "premium," "mid-price," "budget" for price) and test whether binned features improve RF performance. However, RF already handles continuous features well via splits, so the benefit would be marginal.

- **Feature selection via Information Value (IV):** The greedy feature selection algorithm (Algorithm 1) that iteratively adds features based on information value improvement is a clean method. For our project, we could use `sklearn.feature_selection.mutual_info_regression` as a simpler equivalent to identify which of our ~30+ features are actually predictive and prune the rest. This is more directly applicable than IV specifically.

- **Inference as a JOIN / prediction table concept:** The idea of pre-computing predictions and storing them in a lookup table is interesting for UC2. If we pre-compute gameweek predictions for all 700 players and store them, the FastAPI endpoint could serve predictions via simple dict lookup rather than model inference. But this is a performance optimization, not an accuracy improvement.

### 4. Implementation Difficulty

- Feature selection with mutual information: **EASY** (< 1 day) — `sklearn.feature_selection`
- OptBinning for feature discretization: **MEDIUM** (1–2 days) — need to install and learn the library
- Prediction table caching: **EASY** (< 1 day) — simple dict/database cache

### 5. Expected Impact

- Feature selection might improve MAE by 0.5–1 point by removing noise features.
- Discretization and caching would not improve accuracy.
- This paper is primarily about systems engineering, not prediction quality — minimal impact on our core problem.

---

## Cross-Paper Synthesis

### 1. Common Themes

**a) Predict-then-optimize is the dominant paradigm.** Papers 1, 3, and 4 all use ML prediction followed by mathematical optimization (MIP/LP). Paper 2 tries to replace this with RL but achieves weaker results. Our architecture is validated.

**b) Optimization matters as much as (or more than) prediction.** Paper 1 explicitly shows that MIP optimization with mediocre predictions outperforms good predictions with poor optimization. Paper 3's real-world DFS ranking (top 18.4%) is driven by the combination of decent predictions + LP optimization. Our ILP is a major asset.

**c) Position/player-specific modeling helps.** Paper 1 selects different algorithms per position. Paper 3 builds individual models per player. Both find that one-size-fits-all is suboptimal. We already split by position but should consider algorithm selection per position.

**d) Feature engineering with lag/rolling features is universal.** Paper 1 uses N prior game-weeks. Paper 2 uses 90-day rolling averages. Paper 3 uses 1/3/5/7/10-game lags plus momentum features. All agree that recent form features are critical — and we're currently only using 1-year lag features.

**e) Ensembling consistently performs well.** Paper 3's Voting Meta-Model was the most frequently best-performing approach. Simple averaging of 2–3 models is a low-cost, high-return improvement.

**f) Linear optimization (MIP/LP) is fast and effective for fantasy sports.** Papers 1 and 3 both use linear programming for team selection with constraints identical in structure to ours. Our OR-Tools approach is the right choice.

### 2. Priority-Ordered Recommendation List

| Priority | Takeaway | From Paper | Impact | Effort | What It Improves |
|----------|----------|------------|--------|--------|------------------|
| 1 | Add `2_years_past_*` and `3_years_past_*` lag features to season model | Paper 3 | HIGH | EASY | Season MAE (all positions) |
| 2 | Build VotingRegressor ensemble (RF + GBM + Ridge per position) | Paper 3 | HIGH | EASY | Season MAE (1–2 pts) |
| 3 | Build rolling GW features for UC2 (last 3/5/7 GW points) | Papers 1, 2, 3 | HIGH | EASY | GW prediction (UC2) |
| 4 | Add opponent features (FDR) to both season and GW models | Paper 3 | MEDIUM | EASY | Season MAE + GW prediction |
| 5 | Hyperparameter tuning (currently using defaults) | General best practice | MEDIUM | MEDIUM | Season MAE (1–3 pts) |
| 6 | Add momentum features (delta between consecutive periods) | Paper 3 | MEDIUM | EASY | GW prediction + season MAE |
| 7 | Implement two-step validation (train/test/unseen split) | Paper 3 | MEDIUM | EASY | Robustness/explainability |
| 8 | Test different algorithms per position (LR, GBM, ElasticNet) | Papers 1, 3 | MEDIUM | MEDIUM | Season MAE (position-specific) |
| 9 | Feature selection (mutual information or permutation importance) | Papers 3, 4 | LOW-MED | EASY | Season MAE (reduce overfit) |
| 10 | Anomaly detection / target smoothing on training data | Paper 3 | LOW | EASY | Season MAE (noise reduction) |
| 11 | Per-player models for top ~50 premium players | Paper 3 | LOW-MED | MEDIUM | Season MAE for key players |

### 3. What to Ignore

- **Reinforcement Learning for team selection (Paper 2):** Requires GPU infrastructure, millions of training steps, and doesn't guarantee constraint satisfaction. Our ILP is provably optimal under given predictions. RL adds complexity with no clear benefit for a 15-player squad with hard constraints. The 67th-percentile result is not compelling compared to ILP-based approaches achieving top 18.4% (Paper 3).

- **InferDB index-based inference (Paper 4):** Solves a latency problem we don't have. Predicting 700 players takes <1 second with scikit-learn. The supervised discretization is interesting but RF already handles continuous feature splits natively.

- **LSTM for time-series prediction (Paper 1):** LSTMs performed worst in Paper 1's comparison (RMSE of 7.62 for QB vs. 6.89 for Linear Regression). They require careful tuning, more data than we have per player, and sacrifice explainability. Stick with tree-based and linear models.

- **Per-player modeling for all ~700 players (Paper 3):** Paper 3 only modeled 203 players who met strict criteria (100+ appearances over 3 seasons, 30+ games in target season, 18+ min/game). Many FPL players have sparse history. Per-player models for all players would produce unstable estimates for those with limited data.

- **DQN / PPO neural architectures (Paper 2):** Complex deep learning architectures (256→512→1024 layer networks) are overkill for our problem size and violate our explainability requirement.

### 4. Biggest Gap in Our Current Approach

**We are leaving multi-season lag data on the table.** The single most impactful thing we're NOT doing is using our available `2_years_past_*` and `3_years_past_*` features. The data is already in our dataset but completely unused. Every paper reviewed here emphasizes that historical performance over multiple time periods is critical for prediction quality. Paper 3 explicitly tested 3-season vs. 10-season windows and found that both contribute value, with the combination of lag features at different horizons producing the best results.

Adding these features requires zero new data collection — just expanding the feature matrix from `price_now + 1_years_past_*` to `price_now + 1_years_past_* + 2_years_past_* + 3_years_past_*`. Combined with a VotingRegressor ensemble (Priority #2), this should yield the largest single improvement in season MAE across all positions. For players with only 1 year of history, the 2-year and 3-year features will be NaN — RF handles missing values via surrogate splits, or we can impute with 0 (indicating "no prior data").

The second biggest gap is the **complete absence of a trained gameweek model (UC2)**. The current placeholder (using `ep_next` or `points_per_game`) is not a model — it's a fallback. Papers 1 and 3 both show that rolling window features from recent game-weeks produce viable short-term predictions. Building even a simple model using last-N-GW features + FDR + form would be a massive upgrade over the placeholder.
