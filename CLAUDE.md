# CLAUDE.md — Automatic Champion

## What This Project Is

Automatic Champion is a Fantasy Premier League (FPL) squad optimizer — a B.Sc. Software Engineering final project (Afeka College). It uses ML models to predict player performance and integer linear programming (ILP) to build optimal squads under FPL constraints.

**Authors:** Yuval Davidovits, Yuval Garzon
**Deadline:** July 17–20, 2026 (final hand-in + defense)
**Engineering report:** Submitted January 2026

---

## Repo Structure

```
src/              Core library — shared by CLI and backend (DO NOT duplicate logic)
  team_builder.py   ILP optimizer (OR-Tools), prediction, constraint validation
cli/              CLI entry point
  build_team.py     Interactive menu-driven team builder
backend/          FastAPI server + PostgreSQL
  app/
    main.py         FastAPI app, /health endpoint
    database.py     SQLAlchemy engine (PostgreSQL)
    deps.py         Dependency injection (DB session)
    models.py       ORM models (6 tables)
    schemas.py      Pydantic request/response schemas
    routers/
      squad.py      POST /squad/generate endpoint
    services/
      optimizer.py  Backend optimizer (currently greedy — MUST be replaced with ILP from src/)
  scripts/
    seed_db.py      Seed DB from FPL GitHub CSV data
training/         ML training pipelines (run manually, produce .joblib files)
analysis/         One-off analysis, visualization, data prep scripts
tests/            Automated tests
data/
  base/             10 seasons of raw FPL data (2016-17 through 2025-26)
  historical_exports/  Pre-built training CSVs per season
  season_comparisons/  Season delta CSVs
  players_merged_2024-25.csv  Current season player data
models/           Trained .joblib model files (4 basic RF + 4 advanced)
outputs/          Generated squad CSVs/JSONs
visuals/          Generated charts and metrics CSVs
```

## Tech Stack

- **Language:** Python 3.11+
- **ML:** scikit-learn, XGBoost, LightGBM (position-specific models)
- **Optimization:** Google OR-Tools (SCIP/CBC solver) — Integer Linear Programming
- **Backend:** FastAPI + SQLAlchemy + PostgreSQL
- **Frontend:** React + TypeScript (not yet built)
- **Auth:** Firebase (not yet implemented)
- **Data:** pandas, numpy, joblib

## Key Design Rules

1. **`src/` is the single source of truth** for optimization and prediction logic. Both `cli/` and `backend/` must import from `src/`. Never duplicate optimizer code.
2. **ILP, not greedy.** The backend currently has a greedy heuristic in `backend/app/services/optimizer.py` — this MUST be replaced with the ILP optimizer from `src/team_builder.py`.
3. **Explanations are the #1 differentiator.** The report positions transparency/explainability as the core competitive advantage. Every recommendation must include human-readable explanations for why players were selected.
4. **Position-specific models.** There are 4 separate ML models (GK, DEF, MID, FWD). Use `position_model_*.joblib` (basic RandomForest) — the advanced models have broken DEF/MID files (4.8KB each).
5. **FPL constraints are strict:** budget cap (default 100.0), max 3 players per club, position requirements (2GK/5DEF/5MID/3FWD for full squad), valid formations for starting XI.

## Two Core Use Cases

### UC1 — Build Initial Squad (15 players)
- User sets budget, formation, optional locked/banned players
- System predicts season points → ILP optimizer → 15-player squad (11 starters + 4 bench)
- Output includes explanations for why players were selected
- **Status:** Works in CLI. Backend endpoint exists but uses wrong optimizer and has no explanations.

### UC2 — Recommend Weekly Lineup (starting 11 from existing squad)
- User provides their 15-player squad + target gameweek
- System predicts GW points → ILP optimizer → starting XI + bench order + captain
- Output includes explanations + optional transfer suggestion
- **Status:** NOT IMPLEMENTED. This is the biggest gap.

## Database Schema (PostgreSQL)

6 tables: User, Constraint_Set, Team, Player, Season_Team, Season_Team_Player.
Connection: `DATABASE_URL` env var (default: `postgresql+psycopg2://postgres:postgres@localhost:5432/automatic_champion`).
Seed with: `python backend/scripts/seed_db.py`

## Running Things

```bash
# Activate venv
source .venv/bin/activate

# CLI (interactive)
python -m cli.build_team

# CLI (non-interactive)
python -m cli.build_team --budget 100 --formation 4-3-3

# Backend
uvicorn backend.app.main:app --reload --port 5000

# Tests
python -m pytest tests/ -v

# Train models
python -m training.train_advanced_models
python -m training.train_position_models

# Seed database
python backend/scripts/seed_db.py
```

## Test Plan (from report)

| ID | Description | Status |
|----|-------------|--------|
| TC-01 | Build squad with defaults → valid 15 players | Partial (tests 11 only) |
| TC-02 | Build squad with user constraints (locked/banned) | Missing |
| TC-03 | Infeasible constraints → explain conflict | Missing |
| TC-04 | Weekly lineup from valid squad → valid 11 + bench | Missing (UC2 not built) |
| TC-05 | Invalid squad input → block and explain | Missing |
| TC-06 | Missing/stale data → warn and offer options | Missing |
| NFR-01 | 30 requests, average ≤ 20 seconds | Missing (likely passes) |
| NFR-02 | 5 new users, ≥ 80% success without help | Missing (needs user study) |
| NFR-03 | Swap model with dummy → still works | Missing |

---

## Implementation Plan (Phases)

### Phase 1: Foundation Fixes (est. ~8 dev-days)

**Goal:** Unify architecture, fix inconsistencies, make backend match CLI.

- [ ] **1.1 Create `requirements.txt`** — pin all dependencies
- [ ] **1.2 Replace backend greedy optimizer with ILP** — rewrite `backend/app/services/optimizer.py` to import and wrap `src.team_builder.build_team()`. Delete the greedy heuristic entirely.
- [ ] **1.3 Align model files** — standardize on `position_model_*.joblib` (basic RF). The `advanced_model_DEF.joblib` and `advanced_model_MID.joblib` are only 4.8KB and likely broken.
- [ ] **1.4 Extract explanation service** — create `src/explainer.py` by porting `FEATURE_EXPLANATIONS` dict and `get_top_features_for_player/position` from `cli/build_team.py`. Add explanation fields to API response schemas.
- [ ] **1.5 Simplify API contract** — backend should read players from DB/CSV, not receive 500+ players in request body. Add `formation`, `locked_ids`, `banned_ids` to request schema.
- [ ] **1.6 Extract bench builder to `src/`** — move `_build_bench_auto` from `cli/build_team.py` into `src/team_builder.py` so both CLI and backend share it.
- [ ] **1.7 Move DB creds to env var** — `os.environ.get("DATABASE_URL", default)` in `database.py`
- [ ] **1.8 Add solver timeout** — `solver.SetTimeLimit(30_000)` in `src/team_builder.py:_solve_ilp`

### Phase 2: UC2 — Weekly Lineup (est. ~10 dev-days)

**Goal:** Implement the entire missing UC2 flow.

- [ ] **2.1 FPL API data fetcher** — create `src/fpl_api.py`. Wrapper for FPL bootstrap-static API to fetch player form, fixture difficulty. Cache results.
- [ ] **2.2 Gameweek prediction model** — create `src/gameweek_predictor.py`. Pragmatic approach: season model prediction x form factor x fixture difficulty adjustment. No new model training needed.
- [ ] **2.3 Lineup optimizer** — create `src/lineup_optimizer.py`. ILP: pick 11 from 15 maximizing GW points, subject to formation rules. Include captain (2x) and vice-captain.
- [ ] **2.4 Transfer recommender** — create `src/transfer_advisor.py`. For each squad player, compute best available replacement. Return top suggestion.
- [ ] **2.5 UC2 API endpoint** — add `POST /lineup/recommend` in `backend/app/routers/lineup.py`. Request: squad player IDs + formation + gameweek. Response: starters, bench, captain, transfer suggestion, explanations.

### Phase 3: React Frontend (est. ~14 dev-days)

**Goal:** Build the web app from the report. Two pages matching the two use cases.

- [ ] **3.1 Scaffold** — Vite + React + TypeScript in `/frontend/`
- [ ] **3.2 CORS** — add middleware in `backend/app/main.py`
- [ ] **3.3 API client** — `/frontend/src/api/` typed wrappers
- [ ] **3.4 Squad Builder page (UC1)** — budget slider, formation dropdown, locked/banned search, pitch view, explanation panel
- [ ] **3.5 Lineup Advisor page (UC2)** — squad input, recommendation display, transfer suggestion, explanations
- [ ] **3.6 Shared components** — Navbar, ExplanationPanel, PlayerCard, SquadPitch
- [ ] **3.7 Demo polish** — responsive, looks good for defense presentation

### Phase 4: Auth & Security (est. ~4 dev-days)

- [ ] **4.1 Firebase project setup**
- [ ] **4.2 Frontend auth** — AuthContext, Login/Register pages, token in API headers
- [ ] **4.3 Backend token verification** — `backend/app/auth.py` with FastAPI dependency

### Phase 5: Testing (est. ~8 dev-days)

- [ ] **5.1 TC-01** — all 7 formations, full 15-player squad, budget edge cases
- [ ] **5.2 TC-02** — locked/banned constraints
- [ ] **5.3 TC-03** — infeasible constraints → descriptive error
- [ ] **5.4 TC-04** — weekly lineup from valid squad (once UC2 exists)
- [ ] **5.5 TC-05** — invalid input → block and explain
- [ ] **5.6 TC-06** — missing/stale data → warn
- [ ] **5.7 NFR-01** — 30 requests, avg ≤ 20 seconds
- [ ] **5.8 NFR-03** — swap model with DummyModel → system still works
- [ ] **5.9 API integration tests** — FastAPI TestClient with test DB

### Phase 6: Polish & Documentation (est. ~4 dev-days)

- [ ] **6.1 README** — setup, architecture, how to run, API docs
- [ ] **6.2 Docker Compose** — Postgres + backend + frontend
- [ ] **6.3 Backend logging** — Python logging module
- [ ] **6.4 Clean up dead code** — remove greedy optimizer, unused scripts
- [ ] **6.5 Wire Constraint_Set.value_json** — actually read and apply stored constraints

---

## Parallelization (2 students)

```
Student A (backend-focused):  Phase 1 → Phase 2 → Phase 4 backend → Phase 5 backend tests
Student B (frontend-focused): Phase 3 scaffold (week 5) → Phase 3 UI → Phase 4 frontend → Phase 5 frontend tests
Both:                         Phase 6
```

## What Can Be Descoped If Time Is Short

- Transfer recommender (Task 2.4) — nice to have
- Docker Compose (Task 6.2) — run manually for demo
- NFR-02 usability study — do informally with classmates
- Robust optimization / Monte Carlo — report mentions as enhancement, explicitly optional

## What Cannot Be Cut

- UC1 end-to-end via web UI with explanations
- UC2 end-to-end via web UI with explanations
- Firebase auth (or equivalent)
- Tests matching TC-01 through TC-06 and NFR-01/NFR-03
- Explanation system in the API response

## Known Issues

- `backend/app/services/optimizer.py` uses greedy heuristic instead of ILP — MUST replace (Phase 1.2)
- `advanced_model_DEF.joblib` and `advanced_model_MID.joblib` are only 4.8KB — likely broken/degenerate
- `Constraint_Set.value_json` is stored in DB but never read by the optimizer
- Hardcoded DB credentials in `backend/app/database.py`
- No `requirements.txt` or `pyproject.toml` yet
