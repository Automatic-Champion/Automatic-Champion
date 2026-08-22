# Automatic Champion

**Automatic Champion** is a Fantasy Premier League (FPL) squad optimizer. It predicts how many points every player will score with machine-learning models, picks the best legal squad with Integer Linear Programming (ILP), and explains every pick in plain English.

The whole system is one pipeline:

```
   PREDICT                 OPTIMIZE                    EXPLAIN
   ML models score    →    ILP solver picks the   →    each pick gets a
   every player            best legal squad            human-readable reason
   (per position)          (budget, formation,         (feature importance
                            max 3 per club)             and SHAP values)
```

> B.Sc. Software Engineering final project, Afeka College.
> Authors: **Yuval Davidovits**, **Yuval Garzon**. Supervisor: **Yair Even Zohar**.

---

## What It Does

**UC1 — Build a squad (15 players).** You choose a budget, a formation, and optionally lock or ban specific players. The system predicts each player's *season* points, solves the ILP, and returns 11 starters + 4 bench with the total cost, expected points, and an explanation per player.

**UC2 — Weekly lineup advisor (starting 11).** You load a saved 15-player squad and pick a gameweek. The system predicts each player's *next-gameweek* points with CatBoost models, then chooses the best starting XI, bench order, captain, and vice-captain — with a SHAP-based reason for each starter.

Users sign in with Firebase (email/password) and can save up to 10 named squads.

---

## Key Code

The heart of the project is the code below. Each link points at the exact function, with what it does and why it matters.

### Optimization — the ILP solver

| Code | What it does | Why it matters |
|------|--------------|----------------|
| [`_solve_ilp()`](src/team_builder.py#L217-L267) | The core optimization model. One binary variable per player, then the FPL rules as linear constraints: total cost ≤ budget, exact position counts, ≤ 3 players per club, locked players forced in (`x = 1`), banned players forced out (`x = 0`). Objective: `maximize Σ xᵢ · predicted_pointsᵢ`. | This is the mathematical core of the project. It guarantees a **provably optimal** squad, not a "good enough" greedy pick. |
| [`build_full_squad()`](src/team_builder.py#L505-L619) | The UC1 pipeline end to end: load players → predict points → validate constraints → one ILP solve for all 15 → split into starters (by formation) and bench (ordered by predicted points). | The single entry point the backend calls for UC1. |
| [`_build_solver()`](src/team_builder.py#L204-L214) | Creates the OR-Tools solver (SCIP, falling back to CBC) and sets a **30-second time limit**. | The time limit keeps the API responsive even on hard constraint sets. |
| [`FORMATION_COUNTS` / `FULL_SQUAD_COUNTS`](src/team_builder.py#L23-L32) | The official FPL rules as data: 7 valid formations, and the 2 GK / 5 DEF / 5 MID / 3 FWD squad shape. | Every constraint in both optimizers is derived from these two constants. |
| [`_validate_locked_constraints()`](src/team_builder.py#L282-L316) | Checks locked/banned requests *before* the solve — overlapping lock+ban, unknown IDs, locked players over budget, too many locked players per position or per club. | Turns "no solution found" into a specific, human-readable reason (test case TC-03). |
| [`optimize_lineup()`](src/lineup_optimizer.py#L18-L56) | UC2 optimizer. If no formation is given, it solves **all 7 formations** and keeps the highest-scoring one. | Lets the system find the best shape for the squad, not just the one the user guessed. |
| [`_solve_for_formation()`](src/lineup_optimizer.py#L85-L181) | A second, smaller ILP: pick exactly 11 of the 15 players matching the formation, maximizing gameweek points. Then captain = top scorer, vice = second, and `total = Σ starters + captain` (the captain's points count twice). Bench is ordered by points with the goalkeeper always last. | Encodes the real FPL scoring rules for a gameweek, including the captain multiplier. |

### Prediction — the ML models

| Code | What it does | Why it matters |
|------|--------------|----------------|
| [`_predict()`](src/team_builder.py#L167-L201) | Runs the seasonal models — one per position (GK/DEF/MID/FWD) — each on its own selected feature list, and returns a predicted season total for every player. | Produces the objective-function coefficients the ILP maximizes. Garbage in here means a perfectly optimal but useless squad. |
| [`_add_momentum_features()`](src/team_builder.py#L90-L110) | Builds the momentum features with the formula `momentum_X = 1_years_past_X − 2_years_past_X` for points, minutes, ICT index, and goals. | A cheap, strong signal: it separates players who are *improving* from players coasting on an old good season. |
| [`predict_gameweek_points_with_meta()`](src/gameweek_predictor.py#L239-L342) | The V8 weekly predictor. For each player: find the right feature row, run the position's CatBoost model, clip the result to a sane range. Anything it can't predict falls back to `ep_next → points_per_game → season_pred/38`, and it reports exactly how much of the answer came from the fallback. | Keeps UC2 working even when models or data are missing, and the returned metadata becomes the user-facing staleness warning (test case TC-06). |
| [`_rows_by_element()`](src/gameweek_predictor.py#L139-L159) | Picks the correct feature row per player: a row at GW *N* holds what was known *before* GW *N*, so predicting gameweek *G* uses the latest row with `GW < G`. | Prevents data leakage — using a row from the gameweek being predicted would leak the answer into the features. |
| [`BASE_FEATURES` / rolling features](src/gameweek_predictor.py#L62-L117) | The V8 feature list — form, minutes, ICT, expected goals/assists, team strength, and rolling means over 3/5/7 gameweeks — copied verbatim from the training notebook. | Inference features must match training features exactly, or the model silently degrades (serving skew). |
| [`_placeholder_predict()`](src/gameweek_predictor.py#L431-L450) | The fallback chain when the model has no row for a player. | The graceful-degradation path: the app answers instead of erroring. |

### Explanations — the project's main differentiator

| Code | What it does | Why it matters |
|------|--------------|----------------|
| [`_explain_selection_preloaded()`](src/explainer.py#L58-L199) | Explains a UC1 pick by ranking the player's stats **against all other players in the same position** and turning the strongest ones into sentences ("18 goals last season — top 4% of midfielders for goal threat"). | Comparison against peers is what makes a number meaningful. Raw stats don't tell a user *why* the optimizer wanted this player. |
| [`explain_squad()`](src/explainer.py#L231-L266) | Generates explanations for a whole squad in one pass, pre-computing per-position tables instead of re-reading the data per player. | Keeps UC1 well inside the performance requirement (NFR-01). |
| [`_shap_for_rows()`](src/weekly_explainer.py#L315-L338) | Computes CatBoost SHAP values — the exact contribution of each feature to *this* player's prediction — batched per position. | The explanation comes from the model that actually made the prediction, so it can never contradict it. |
| [`_reasons_from_shap()`](src/weekly_explainer.py#L255-L283) | Turns raw SHAP numbers into readable reasons: collapses a concept to its strongest time window, drops empty/zero stats, keeps only the features that pushed the prediction **up**, strongest first. | Bridges model output and user trust — this is what a user actually reads next to each player. |
| [`explain_weekly_squad()`](src/weekly_explainer.py#L340-L384) | The UC2 explanation entry point, with a feature-importance fallback if SHAP fails. | Explanations never disappear, even in a degraded run. |

### Backend and API

| Code | What it does | Why it matters |
|------|--------------|----------------|
| [`generate_squad()`](backend/app/routers/squad.py#L39-L83) | `POST /squad/generate` — validates the request, calls the optimizer, attaches explanations, maps infeasible constraints to a `400` with a clear message. | The UC1 endpoint the web app calls. |
| [`recommend_lineup()`](backend/app/routers/lineup.py#L20-L109) | `POST /lineup/recommend` — predicts the gameweek, optimizes the XI, and returns starters, bench, captain/vice, and any data `warnings`. | The UC2 endpoint the web app calls. |
| [`generate_optimal_squad()`](backend/app/services/optimizer.py) | A thin wrapper over `src/team_builder.build_full_squad()` that converts library errors into one `OptimizationError`. | Keeps `src/` the single source of truth — the backend holds no optimization logic of its own. |
| [`get_current_user()`](backend/app/auth.py#L44-L80) | FastAPI dependency that verifies the Firebase ID token on every protected request. | Every squad is scoped to the user who created it. |

---

## Tech Stack

- **Python 3.11+** — core library, backend, and training
- **ML (seasonal, UC1):** one model per position, chosen by benchmark — GK: XGBoost, DEF: ElasticNet, MID: Ridge, FWD: LightGBM (`models/position_model_{1..4}.joblib`)
- **ML (weekly, UC2):** CatBoost per position, V8, tuned with Optuna (`Weekly Model/production/models/`)
- **Optimization:** Google OR-Tools (SCIP / CBC) — Integer Linear Programming
- **Backend:** FastAPI + SQLAlchemy + PostgreSQL + Firebase Admin SDK
- **Frontend:** React + TypeScript + Vite + Tailwind v4 + shadcn/ui
- **Data:** pandas, numpy, joblib

---

## Repository Structure

```
src/             Core library — optimizer, predictors, explainers (single source of truth)
backend/         FastAPI server: routers, services, ORM models, Firebase auth
frontend/        React + TypeScript web app (the deliverable UI)
Weekly Model/    V8 CatBoost models, training notebook, runtime feature table
training/        Training + evaluation pipelines for the seasonal models
tests/           pytest suite (~116 tests)
data/            FPL data, including the player table the optimizer reads
docs/            Model reports, test results, system flow
cli/             Interactive CLI prototype (reference only, not the deliverable)
```

---

## Setup

```bash
git clone https://github.com/Automatic-Champion/Automatic-Champion.git
cd Automatic-Champion

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install && cd ..

docker compose up -d            # PostgreSQL on localhost:5432

python -c "import sys; sys.path.insert(0,'backend'); from app.database import engine, Base; from app.models import *; Base.metadata.create_all(bind=engine)"
python backend/scripts/seed_db.py
```

The backend needs a Firebase service-account JSON (kept **outside** the repo) via `GOOGLE_APPLICATION_CREDENTIALS`. Both that variable and `DATABASE_URL` can live in a gitignored `.env` at the repo root.

## Running

```bash
uvicorn backend.app.main:app --reload --port 8000
```

```bash
cd frontend && npm run dev
```

Then open <http://localhost:5173>. API docs: <http://localhost:8000/docs>.

## API

| Method | Path | Auth | Description |
|--------|------|:----:|-------------|
| GET | `/health` | No | Liveness + database check |
| GET | `/players` | No | All selectable players |
| POST | `/squad/generate` | Yes | UC1 — build an optimal 15-player squad |
| POST | `/lineup/recommend` | Yes | UC2 — recommend the starting XI for a gameweek |
| GET/POST | `/squads` | Yes | List / save named squads (max 10 per user) |
| GET/DELETE | `/squads/{id}` | Yes | Load or delete one saved squad |

## Tests

```bash
python -m pytest tests/ -v
```

About 116 tests covering the report's test plan: squad building and constraints (TC-01, TC-02), infeasible constraints (TC-03), weekly lineup (TC-04), invalid input (TC-05), stale-data warnings (TC-06), performance (NFR-01, 30 requests, measured 0.87 s average), and model-swap robustness (NFR-03).

## Model Results

V8 weekly models, held-out 2024-25 season (never seen in training):

| Position | Test MAE |
|----------|----------|
| GK | 0.73 |
| DEF | 1.03 |
| MID | 1.00 |
| FWD | 1.15 |
| **Average** | **0.98** |

70.5% of predictions land within ±1 point, 85.2% within ±2, 91.8% within ±3.
Full methodology is in [`Weekly Model/README.md`](Weekly%20Model/README.md); the seasonal models are documented in [`docs/ml_model_report.md`](docs/ml_model_report.md).
