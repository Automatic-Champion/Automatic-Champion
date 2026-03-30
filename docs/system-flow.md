# Automatic Champion — End-to-End System Documentation

## Part 1: Data Foundation

### What data does the system use?

The system relies on three data sources:

1. **A CSV file** (`data/players_merged_2024-25.csv`) — 807 player rows covering the 2024-25 Premier League season. This is the primary data source for UC1 (squad building). It contains current-season stats and up to 3 seasons of historical stats per player.

2. **The live FPL API** (`https://fantasy.premierleague.com/api`) — used in UC2 (lineup recommendation) to get real-time player data like expected points (`ep_next`), current form, points per game, and fixture information. Accessed via `src/fpl_api.py`.

3. **Trained ML models** (`.joblib` files in `models/`) — four position-specific Random Forest models that predict season-long total points for each player. The system uses the "basic" models:
   - `position_model_1.joblib` → GK (goalkeepers)
   - `position_model_2.joblib` → DEF (defenders)
   - `position_model_3.joblib` → MID (midfielders)
   - `position_model_4.joblib` → FWD (forwards)

   There are also four "advanced" models (`advanced_model_*.joblib`), but the DEF and MID ones are only 4.8KB and are likely broken, so the system uses the basic position models.

### What's in the CSV?

The CSV header row has 76 columns. Each row is one player. Key columns:

- **Identity:** `id` (unique player ID), `first_name`, `second_name`, `team_name`, `element_type` (position code: 1=GK, 2=DEF, 3=MID, 4=FWD), `position` (string label)
- **Current season:** `total_points`, `price_now` (cost in tenths — e.g., 77 means £7.7m)
- **1-year historical stats** (prefix `1_years_past_`): `goals_scored`, `assists`, `total_points`, `minutes`, `goals_conceded`, `creativity`, `influence`, `threat`, `bonus`, `bps`, `ict_index`, `clean_sheets`, `red_cards`, `yellow_cards`, `selected_by_percent`, `now_cost`, `element_type`, `gw_games_played`, `gw_minutes_per_game`, `gw_penalties_missed`, `gw_penalties_saved`, `gw_own_goals`, `gw_saves`
- **2-year and 3-year historical stats** (same fields with prefixes `2_years_past_` and `3_years_past_`)

For example, Gabriel Jesus (row 2) is `id=2`, `team_name=Arsenal`, `element_type=4` (FWD), `price_now=65` (£6.5m), `total_points=42`, with `1_years_past_goals_scored=4`, `1_years_past_total_points=85`, etc.

### What are the trained models?

Each model is a scikit-learn RandomForest regressor (loaded via `joblib.load()`). They predict a single number: **expected total season points** for a player. There is one model per position because different stats matter differently for each role (e.g., saves matter for GKs, goals matter for FWDs).

### What features do the models use?

Features are determined dynamically by `_build_feature_cols()` (`src/team_builder.py:58-81`). The function selects:

1. `price_now` — the player's current price (always included)
2. Every column that starts with `1_years_past_` and is numeric — these are last season's stats

This means the feature set is approximately 23 columns: `price_now` plus the 22 `1_years_past_*` columns (goals_scored, assists, total_points, minutes, goals_conceded, creativity, influence, threat, bonus, bps, ict_index, clean_sheets, red_cards, yellow_cards, selected_by_percent, now_cost, element_type, gw_games_played, gw_minutes_per_game, gw_penalties_missed, gw_penalties_saved, gw_own_goals, gw_saves).

Note: only `1_years_past_*` columns are used as features — the `2_years_past_*` and `3_years_past_*` columns exist in the CSV but are not fed to the prediction models.

### How is the CSV loaded and preprocessed?

The function `_prepare_player_df()` (`src/team_builder.py:260-311`) does this step by step:

1. **Load:** Reads the CSV into a pandas DataFrame (`pd.read_csv`).
2. **Validate columns:** Checks that `id`, `team_name`, `first_name`, `second_name`, `element_type`, and `price_now` all exist.
3. **Drop bad rows:** Removes rows where `id` or `team_name` is missing or blank.
4. **Normalize ID:** Converts `id` to float → int → string (e.g., `2.0` → `"2"`).
5. **Build name:** Concatenates `first_name` and `second_name` with a space, collapses multiple spaces, strips whitespace. For example, "Gabriel" + "Fernando de Jesus" → "Gabriel Fernando de Jesus".
6. **Clean team:** Strips whitespace from `team_name`.
7. **Filter positions:** Removes any row where `element_type` is not in {1, 2, 3, 4}. Records how many were dropped and what invalid values were seen.
8. **Map positions:** Converts numeric `element_type` to string position via `POSITION_MAP` (1→"GK", 2→"DEF", 3→"MID", 4→"FWD").
9. **Build features:** Calls `_build_feature_cols()` to identify the feature columns. Coerces them all to numeric.
10. **Process cost:** Converts `price_now` to numeric, creates `cost_int` (rounded integer, in tenths) and `cost` (divided by 10, so `77` → `7.7`).
11. **Predict:** If `pred_mode="models"`, calls `_predict()` to generate predicted season points for every player using the ML models. If `pred_mode="stub"`, expects a `pred` column already in the CSV.
12. **Return:** Returns the prepared DataFrame and a metadata dict recording how many rows were dropped.

---

## Part 2: UC1 — Build Squad (Full Flow)

### 1. Request arrives

The user sends a `POST` request to `/squad/generate`. The request body is a JSON object matching the `SquadGenerateRequest` schema (`backend/app/schemas.py:103-107`):

```json
{
  "budget": 100.0,
  "formation": "4-3-3",
  "locked_ids": [283, 318],
  "banned_ids": [10]
}
```

- `budget` — total squad budget in millions (default 100.0, min 1.0, max 500.0)
- `formation` — one of 7 valid formations: "4-3-3", "4-4-2", "3-4-3", "3-5-2", "4-5-1", "5-3-2", "5-4-1" (default "4-3-3")
- `locked_ids` — player IDs that must be included in the squad (default empty)
- `banned_ids` — player IDs that must be excluded (default empty)

### 2. Router handling

The request hits `generate_squad()` in `backend/app/routers/squad.py:38-79`. The router:

1. Converts `locked_ids` and `banned_ids` from lists of ints to sets of strings (or `None` if empty).
2. Calls `generate_optimal_squad()` from the optimizer service, passing `budget`, `formation`, `DATA_PATH` (`"data/players_merged_2024-25.csv"`), `MODELS_DIR` (`"models"`), `locked_ids`, and `banned_ids`.
3. If optimization fails, catches the error and returns HTTP 400 with the error message.
4. Calls `explain_squad()` from `src/explainer.py` for all 15 players (starters + bench). If explanation generation fails, it logs a warning and continues with empty explanations — explanations are non-fatal.
5. Assembles the response by mapping each player dict to a `SquadPlayerResponse` with their explanations attached.

### 3. Optimizer wrapper

`generate_optimal_squad()` in `backend/app/services/optimizer.py:12-31` is a thin wrapper. It calls `build_full_squad()` from `src/team_builder.py` with the exact same parameters. If any exception occurs (ValueError, FileNotFoundError, ImportError, RuntimeError), it wraps it in an `OptimizationError` and re-raises.

### 4. Player data loading

`build_full_squad()` (`src/team_builder.py:445-559`) calls `_prepare_player_df()` (described in Part 1 above) to load the CSV, clean the data, build features, and generate predictions. It then converts each DataFrame row into a `PlayerRecord` dataclass (`src/team_builder.py:478-490`), which bundles `idx`, `player_id`, `name`, `team`, `position`, `cost_int`, `cost`, and `pred` into an immutable object.

### 5. Prediction

When `pred_mode="models"` (the default), `_prepare_player_df()` calls `_predict()` (`src/team_builder.py:119-142`).

**Step by step:**

1. **Load models:** `_load_models()` (`src/team_builder.py:105-116`) loads all four `.joblib` files from the models directory. Each file contains a trained scikit-learn model.

2. **Iterate by position:** For each position (GK, DEF, MID, FWD), the function filters the DataFrame to players of that position, extracts the feature columns (e.g., `price_now`, `1_years_past_goals_scored`, `1_years_past_assists`, ...), and calls `model.predict(subset)`.

3. **Model output:** Each model's `.predict()` returns an array of floats — one per player — representing the predicted total season points. For example, if Salah has `price_now=127`, `1_years_past_total_points=228`, `1_years_past_goals_scored=18`, etc., the MID model might output `190.5` predicted season points.

4. **Fallback:** If a model has `predict_proba` instead of `predict` (classification model), it uses the probability of class 1. However, the current models are regressors with `.predict()`. If prediction produces any NaN values, the function raises a `ValueError`.

**How features are built** (`_build_feature_cols`, `src/team_builder.py:58-81`):
1. Finds all columns starting with `1_years_past_`.
2. Coerces them to numeric (non-numeric values become NaN).
3. Filters to keep only columns that are actually numeric after coercion.
4. Returns `["price_now"] + [all numeric 1_years_past_* columns]`.

### 6. ILP Optimization

**What is ILP?** Integer Linear Programming is a mathematical optimization technique where you maximize (or minimize) a linear function subject to linear constraints, and the decision variables must be integers. In this case, each variable is binary (0 or 1): "include this player or not."

**The solver:** `_build_solver()` (`src/team_builder.py:145-155`) creates an OR-Tools solver. It first tries SCIP (a powerful open-source solver), falling back to CBC if SCIP is unavailable. The timeout is set to 30 seconds (`solver.SetTimeLimit(30_000)`).

**Decision variables:** For each of the ~807 players in the dataset, there is one boolean variable `x_i` ∈ {0, 1}. If `x_i = 1`, that player is selected for the 15-player squad.

**Objective function:** Maximize the sum of predicted season points for all selected players:
```
Maximize Σ (x_i × pred_i) for all players i
```

**Constraints** (all defined in `_solve_ilp`, `src/team_builder.py:158-208`):

1. **Budget constraint** (line 172-174): The total cost of selected players must not exceed the budget.
   ```
   Σ (x_i × cost_int_i) ≤ budget_int
   ```
   Costs are in tenths of millions to avoid floating-point issues (budget of £100m = `budget_int` of 1000).

2. **Position count constraints** (lines 177-180): For a full squad, exactly 2 GK, 5 DEF, 5 MID, 3 FWD must be selected (defined in `FULL_SQUAD_COUNTS`).
   ```
   Σ (x_i for GK players) == 2
   Σ (x_i for DEF players) == 5
   Σ (x_i for MID players) == 5
   Σ (x_i for FWD players) == 3
   ```

3. **Club limit constraint** (lines 183-189): At most 3 players from any single club.
   ```
   For each club: Σ (x_i for club's players) ≤ 3
   ```

4. **Locked player constraints** (lines 192-194): If a player is in `locked_ids`, force `x_i = 1`.

5. **Banned player constraints** (lines 195-196): If a player is in `banned_ids`, force `x_i = 0`.

**How starters + bench are handled in a single solve:** The ILP selects all 15 players at once using `FULL_SQUAD_COUNTS` (2 GK, 5 DEF, 5 MID, 3 FWD). The split into 11 starters and 4 bench happens *after* the ILP solve, not during it.

**What happens on timeout:** If the solver can't find an optimal solution within 30 seconds, it returns the best feasible solution found so far (status `FEASIBLE`). If no feasible solution exists at all, it raises `ValueError("No feasible solution found for the given constraints")`.

**Solution extraction** (lines 205-208): After solving, the code iterates over all players and collects those where `x_i.solution_value() > 0.5` (i.e., selected). It verifies the count matches the expected total (15 for full squad).

### 7. Bench assignment

After the ILP returns 15 chosen players, `build_full_squad()` splits them into starters and bench (`src/team_builder.py:513-531`):

1. **Group by position:** For each position (GK, DEF, MID, FWD), sort the selected players of that position by predicted points descending.
2. **Assign starters:** Take the top N players per position according to the requested formation. For example, in 4-3-3: top 1 GK, top 4 DEF, top 3 MID, top 3 FWD become starters.
3. **Assign bench:** The remaining players (1 GK, 1 DEF, 2 MID, 0 FWD in a 4-3-3) become the bench.
4. **Sort starters** by position order (GK → DEF → MID → FWD), then alphabetically by name.
5. **Sort bench** by predicted points descending — best substitute first.

The `total_pred` in the response is the sum of starters' predictions only (not bench).

### 8. Explanations

Explanations are generated by `explain_squad()` in `src/explainer.py:181-224`. The router calls this after the squad is built (`backend/app/routers/squad.py:57-67`).

**How it works:**

1. **Load data:** Reads the same CSV file into a DataFrame.
2. **Load models:** For each position present in the squad, loads the corresponding `.joblib` model (loading each model only once).
3. **For each player,** calls `_explain_selection_preloaded()` (`src/explainer.py:82-136`):

**Walk-through of `_explain_selection_preloaded` for one player** (say Gabriel Jesus, ID="2", position="FWD"):

1. Get the FWD model from the pre-loaded models dict.
2. Check if the model has `feature_importances_` (a property of tree-based models like Random Forest). If not, return empty — no explanation possible.
3. Find the player's row in the DataFrame by matching `id == "2"`.
4. Get the feature column names — either from `model.feature_names_in_` (if the model stores them) or by calling `_build_feature_cols()`.
5. Get the model's `feature_importances_` array — one importance value per feature, summing to 1.0. These represent how much each feature contributed to the model's overall predictions (not just this player's prediction).
6. Pair each feature with its importance, sort descending.
7. Take the top 3 (`top_k=3`). For each:
   - Get the player's actual value for that feature (e.g., `1_years_past_goals_scored = 4`).
   - Generate a human-readable explanation via `_explain_feature()` (`src/explainer.py:62-79`).

**How `_explain_feature` works:**
- For `price_now`: returns a fixed string about cost/budget fit.
- For `1_years_past_*` columns: strips the prefix to get the base stat name (e.g., `1_years_past_goals_scored` → `goals_scored`), looks it up in `FEATURE_EXPLANATIONS` (a dict mapping stat names to plain-English descriptions), and returns something like: `"Last season goals scored: 4 — Goal threat: more goals usually means more points."`
- If the feature isn't recognized, returns a generic fallback message.

The result for each player is a list of dicts like:
```json
[
  {"feature": "1_years_past_total_points", "value": 85, "importance": 0.32, "explanation": "Last season total points: 85 — Past points: shows how productive he was over a full season."},
  {"feature": "price_now", "value": 65, "importance": 0.18, "explanation": "Cost: a good fit for the budget while keeping quality high."},
  {"feature": "1_years_past_minutes", "value": 1470, "importance": 0.12, "explanation": "Last season minutes: 1470 — Reliability: plays regularly, which increases chances of steady points."}
]
```

### 9. Response construction

The router (`squad.py:69-79`) assembles a `SquadGenerateResponse` containing:

- `formation` — the formation string (e.g., "4-3-3")
- `budget` — the requested budget
- `total_cost` — actual cost of all 15 players (e.g., 99.3)
- `total_predicted_points` — sum of predicted points for starters only
- `players` — list of 11 starter `SquadPlayerResponse` objects, each with `id`, `name`, `team`, `position`, `cost`, `predicted_points`, `is_starter=true`, `bench_order=null`, and `explanations`
- `bench` — list of 4 bench `SquadPlayerResponse` objects, each with `is_starter=false` and `bench_order` (1-4)

---

## Part 3: UC2 — Recommend Lineup (Full Flow)

### 1. Request arrives

The user sends a `POST` request to `/lineup/recommend`. The request body matches `LineupRecommendRequest` (`backend/app/schemas.py:143-146`):

```json
{
  "squad": [
    {"id": "2", "name": "Gabriel Fernando de Jesus", "position": "FWD", "team": "Arsenal", "cost": 6.5, "pred": 85.0},
    "... 14 more players ..."
  ],
  "gameweek": null,
  "formation": null
}
```

- `squad` — exactly 15 player objects, each with `id`, `name`, `position`, `team`, `cost`, `pred` (season prediction)
- `gameweek` — optional; if null, the system auto-detects the current gameweek from the FPL API
- `formation` — optional; if null, the system tries all 7 formations and picks the best

### 2. Router handling

`recommend_lineup()` in `backend/app/routers/lineup.py:21-95`:

1. Converts the Pydantic models to plain dicts (`p.model_dump()` for each player).
2. If `gameweek` is null, calls `get_current_gameweek()` from `src/fpl_api.py` to determine it.
3. Calls `predict_gameweek_points()` to get per-player GW point predictions.
4. Calls `optimize_lineup()` to select the best 11.
5. Generates explanations via `explain_squad()` (same as UC1, non-fatal).
6. Builds the response objects and returns them.

### 3. FPL API fetch

The FPL API wrapper lives in `src/fpl_api.py`.

**Endpoints called:**
- `https://fantasy.premierleague.com/api/bootstrap-static/` — the master endpoint containing all player data (elements), team data, and gameweek events. Called by `fetch_bootstrap()`.
- `https://fantasy.premierleague.com/api/fixtures/` — all fixtures for the season. Called by `fetch_fixtures()`.

**What data comes back from bootstrap-static:**
- `events` — list of gameweek objects, each with `id`, `is_current`, `is_next`, etc.
- `elements` — list of all ~800 players with `id`, `web_name`, `first_name`, `second_name`, `element_type`, `team`, `form`, `points_per_game`, `now_cost`, `ep_next` (expected points next GW), `status`, `chance_of_playing_next_round`, `total_points`, `minutes`.
- `teams` — list of 20 teams with strength ratings.

**Caching:** Responses are cached in a module-level dict `_cache` protected by a threading lock (`src/fpl_api.py:39-62`). Each entry stores a timestamp and the data. The TTL is 300 seconds (5 minutes). On cache hit within TTL, the cached data is returned immediately. On cache miss or expiry, a fresh HTTP request is made.

**If the API is down:** The `_fetch_json()` function raises `FPLAPIError` on HTTP errors, URL errors, or malformed JSON. The gameweek predictor catches this and falls back to season predictions (see below).

### 4. Gameweek prediction

`predict_gameweek_points()` in `src/gameweek_predictor.py:32-92` predicts how many points each of the 15 squad players will score in the upcoming gameweek.

**How the current gameweek is determined:** `get_current_gameweek()` (`src/fpl_api.py:113-126`) iterates through the `events` list from bootstrap-static. It first looks for the event where `is_current=True`. If none is current (e.g., between gameweeks), it falls back to the event where `is_next=True`.

**The fallback chain for each player:**

1. **Try to fetch FPL data.** If the API call fails entirely, *all* players get fallback predictions (`pred / 38.0`).

2. **Match the squad player to an FPL API player by name** (since the squad comes from the CSV with full names, but the FPL API uses different name fields). The matching function `_match_fpl_player()` (`src/gameweek_predictor.py:111-175`) tries 4 strategies in order:

   - **Strategy 1 — web_name word-boundary match:** For each FPL player, check if their `web_name` (e.g., "Salah") appears as a whole word in the squad player's full name (e.g., "Mohamed Salah"). Uses `_word_boundary_match()` which splits on spaces/hyphens and checks for exact word matches.

   - **Strategy 2 — last name substring:** Takes the last word of the squad player's name and checks if it's a substring of the FPL player's `second_name`. Requires length > 2 to avoid false matches on short names.

   - **Strategy 3 — full name substring:** Checks if the squad player's full name is a substring of the FPL full name (`first_name + second_name`), or vice versa.

   - **Strategy 4 — token overlap:** Normalizes both names (lowercase, remove dots/hyphens), then checks if at least 2 tokens are shared AND the positions match.

   If multiple candidates match in any strategy, `_pick_best()` prefers same-position matches.

3. **Once matched, apply the prediction priority:**
   - **`ep_next`** (FPL's official "expected points next gameweek" estimate) — used if available and not None. This is the best signal.
   - **`points_per_game`** — used if `ep_next` is unavailable but `points_per_game > 0`.
   - **`pred / 38.0`** — season prediction divided by 38 gameweeks. This is the last resort.

4. **If no FPL match is found** for a player, they also get `pred / 38.0`.

The output is a dict mapping player ID (string) → predicted GW points (float). For example: `{"2": 3.8, "283": 5.2, ...}`.

### 5. Lineup optimization

`optimize_lineup()` in `src/lineup_optimizer.py:18-55` selects the best starting 11 from the 15-player squad.

**What's different from UC1:** UC1's ILP selects 15 players from ~807 candidates with budget and club constraints. UC2's ILP selects 11 from exactly 15 players — there's no budget constraint (players are already owned) and no club constraint (the squad already satisfies it). The only constraints are position counts per the formation.

**How formations are handled** (lines 44-55): If `formation` is `None`, the optimizer tries all 7 valid formations (4-3-3, 4-4-2, 3-4-3, 3-5-2, 4-5-1, 5-3-2, 5-4-1) and keeps the one that produces the highest `total_gw_points`. If a specific formation is provided, only that one is tried.

**The ILP for a single formation** (`_solve_for_formation`, lines 85-181):

- **Decision variables:** 15 boolean variables `x_0` through `x_14`, one per squad player.
- **Constraints:**
  - Exactly 11 starters: `Σ x_i == 11`
  - Position counts match the formation: e.g., for 4-3-3: `Σ x_i (GK) == 1`, `Σ x_i (DEF) == 4`, `Σ x_i (MID) == 3`, `Σ x_i (FWD) == 3`
- **Objective:** Maximize `Σ (x_i × gw_points_i)`

**Captain selection** (lines 127-132): After solving, the starters are sorted by predicted GW points descending. The player with the highest prediction becomes captain, the second highest becomes vice-captain. The captain's points are doubled in the total (line 135): `total = sum(starter_points) + captain_points`.

**Bench ordering** (lines 157-161): Bench players are split into GK and outfield. Outfield bench players are sorted by predicted GW points descending. GK goes last. This matches FPL's auto-sub rules (outfield subs are prioritized over the backup GK). The bench order is 1-based (1 = first sub, ..., 4 = backup GK).

### 6. Explanations

Explanations in UC2 use the same `explain_squad()` function from `src/explainer.py` as UC1. The lineup router (`lineup.py:46-58`) builds a player list with `id` and `position` for all 15 squad members and passes it to `explain_squad()`.

The explanations are identical in nature — top-3 feature importances from the season prediction model with human-readable text. They describe why the model rates a player highly in general, not why they were specifically picked for *this* gameweek. The explanations are attached to both starters and bench players in the response.

### 7. Response construction

The router (`lineup.py:87-95`) returns a `LineupRecommendResponse` containing:

- `formation` — the selected formation (e.g., "3-4-3")
- `gameweek` — the resolved gameweek number (e.g., 30)
- `captain_id` — ID of the captain
- `vice_captain_id` — ID of the vice-captain
- `total_gw_points` — sum of starters' GW points + captain's GW points (captain counted twice)
- `starters` — list of 11 `LineupStarterResponse` objects with `id`, `name`, `position`, `team`, `gw_points`, `is_captain`, `is_vice_captain`, `explanations`
- `bench` — list of 4 `LineupBenchResponse` objects with `id`, `name`, `position`, `team`, `gw_points`, `bench_order`, `explanations`

---

## Part 4: System Architecture Diagram

```
                            ┌────────────────────────────────────────────┐
                            │              User / Client                 │
                            └──────────┬─────────────┬──────────────────┘
                                       │             │
                              UC1: POST            UC2: POST
                             /squad/generate      /lineup/recommend
                                       │             │
                            ┌──────────▼─────────────▼──────────────────┐
                            │         FastAPI  (backend/app/main.py)     │
                            │                                            │
                            │  /health   squad_router   lineup_router    │
                            └──────┬─────────┬──────────────┬───────────┘
                                   │         │              │
                            ┌──────┘    ┌────▼────┐    ┌────▼──────────────────┐
                            │           │ squad.py│    │     lineup.py         │
                            │           └────┬────┘    └──┬───────┬───────┬────┘
                            │                │            │       │       │
                            │         ┌──────▼──────┐     │       │       │
                            │         │optimizer.py │     │       │       │
                            │         │ (thin wrap) │     │       │       │
                            │         └──────┬──────┘     │       │       │
                            │                │            │       │       │
           ┌────────────────┼────────────────┼────────────┼───────┼───────┼──────┐
           │                │          src/  (core library)                       │
           │                │                │            │       │       │       │
           │   config.py ───┤         ┌──────▼──────┐     │       │       │       │
           │  DATA_PATH     │         │team_builder │     │       │       │       │
           │  MODELS_DIR    │         │build_full_  │     │       │       │       │
           │                │         │squad()      │     │       │       │       │
           │                │         │_predict()   │     │       │       │       │
           │                │         │_solve_ilp() │     │       │       │       │
           │                │         └──────┬──────┘     │       │       │       │
           │                │                │       ┌────▼────┐  │       │       │
           │                │                │       │fpl_api  │  │       │       │
           │                │                │       │fetch_   │  │       │       │
           │                │                │       │bootstrap│  │       │       │
           │                │                │       └────┬────┘  │       │       │
           │                │                │            │  ┌────▼─────┐ │       │
           │                │                │            │  │gameweek_ │ │       │
           │                │                │            │  │predictor │ │       │
           │                │                │            │  │predict_  │ │       │
           │                │                │            │  │gw_points │ │       │
           │                │                │            │  └──────────┘ │       │
           │                │                │            │        ┌──────▼─────┐ │
           │                │                │            │        │lineup_     │ │
           │                │                │            │        │optimizer   │ │
           │                │                │            │        │optimize_   │ │
           │                │                │            │        │lineup()    │ │
           │                │                │            │        └────────────┘ │
           │                │         ┌──────▼──────┐     │                       │
           │                │         │ explainer   │◄────┘  (both UC1 & UC2)     │
           │                │         │explain_squad│                             │
           │                │         └─────────────┘                             │
           └────────────────┼────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼──────────────────┐
              │             │                  │
       ┌──────▼──────┐ ┌───▼────────┐ ┌───────▼──────────┐
       │   CSV file   │ │  .joblib   │ │   FPL API        │
       │ players_     │ │  models    │ │ bootstrap-static │
       │ merged_      │ │ (4 files)  │ │ fixtures         │
       │ 2024-25.csv  │ │            │ │ (live HTTP)      │
       └──────────────┘ └────────────┘ └──────────────────┘
```

**UC1 path:** User → squad.py → optimizer.py → team_builder.build_full_squad() → CSV + models → explainer → Response

**UC2 path:** User → lineup.py → fpl_api (live data) → gameweek_predictor → lineup_optimizer → explainer → Response

---

## Part 5: Key Design Decisions

### Why ILP instead of a greedy algorithm?

A greedy algorithm picks the best available player one at a time. This can get stuck in suboptimal solutions — for example, picking an expensive star player early might leave too little budget for other positions, resulting in a worse overall squad. ILP considers all 807 players simultaneously and guarantees the mathematically optimal combination subject to all constraints. The solution maximizes total predicted points across the entire squad, not just position-by-position.

### Why a single ILP solve for 15 players instead of solving starters and bench separately?

If you solved starters first and then filled the bench with the best remaining players, the bench selections would be constrained by what the starter solve left behind. By solving for all 15 at once with `FULL_SQUAD_COUNTS` (2 GK, 5 DEF, 5 MID, 3 FWD), the ILP can make global trade-offs — for example, slightly downgrading a 5th defender to get a much better 5th midfielder, even if that midfielder ends up on the bench. The starters/bench split is a simple post-processing step (pick the highest-predicted players per position for the requested formation).

### Why position-specific models instead of one model for all players?

Different positions score points in fundamentally different ways in FPL. Goalkeepers earn points from saves and clean sheets. Defenders earn from clean sheets and occasional goals. Midfielders and forwards earn primarily from goals and assists, but midfielders get more points per goal. A single model would conflate these patterns. Four separate models let each one learn the statistics that matter most for its position — saves matter for GKs but are irrelevant for FWDs.

### Why is the gameweek predictor a placeholder? What's the plan for the real one?

The current predictor (`placeholder-v1`) uses FPL's own `ep_next` field as the primary signal, falling back to `points_per_game` or `pred/38`. This is a reasonable baseline but doesn't incorporate fixture difficulty, opponent strength, home/away advantage, injury status, or recent form trends. The partner student is building a real ML model for gameweek prediction. The contract is defined: `predict_gameweek_points(squad, gameweek)` → `dict[str, float]`. The placeholder can be swapped out by implementing the same function signature.

### Why does `src/` exist separately from the backend?

`src/` contains the core logic (optimization, prediction, explanation) as a pure Python library with no web framework dependencies. The backend's `services/optimizer.py` is a thin wrapper that just calls into `src/`. This separation means: (1) the CLI tools can import from `src/` directly without starting the web server; (2) tests can test the core logic independently of FastAPI; (3) if the backend framework changed, only the thin wrapper layer needs updating.
