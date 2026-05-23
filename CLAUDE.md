# CLAUDE.md — Automatic Champion

## What This Project Is

Automatic Champion is a Fantasy Premier League (FPL) squad optimizer — a B.Sc. Software Engineering final project (Afeka College). It uses ML models to predict player performance and integer linear programming (ILP) to build optimal squads under FPL constraints.

**Authors:** Yuval Davidovits, Yuval Garzon
**Deadline:** July 17–20, 2026 (final hand-in + defense)
**Engineering report:** Submitted January 2026

---

## Repo Structure

```
src/              Core library — shared by backend and frontend (DO NOT duplicate logic)
  team_builder.py   ILP optimizer (OR-Tools), prediction, constraint validation, full squad builder
  explainer.py      Explanation service — feature importance with human-readable text
  fpl_api.py        FPL API wrapper (bootstrap-static, fixtures, caching)
  gameweek_predictor.py  GW points predictor (placeholder-v1, uses FPL ep_next/ppg fallback)
  lineup_optimizer.py    Lineup optimizer — ILP picks best 11 from 15, captain/VC, bench order
cli/              CLI tools
  build_team.py     Interactive menu-driven team builder (early prototype)
  test_backend.py   Full interactive CLI to test backend API (UC1 + UC2)
backend/          FastAPI server + PostgreSQL
  app/
    main.py         FastAPI app, /health endpoint, router registration
    database.py     SQLAlchemy engine (PostgreSQL)
    deps.py         Dependency injection (DB session)
    models.py       ORM models (6 tables)
    schemas.py      Pydantic request/response schemas (UC1 + UC2)
    routers/
      squad.py      POST /squad/generate endpoint (UC1)
      lineup.py     POST /lineup/recommend endpoint (UC2)
    services/
      optimizer.py  Thin wrapper around src.team_builder.build_full_squad()
  scripts/
    seed_db.py      Seed DB from FPL GitHub CSV data
training/         ML training pipelines (run manually, produce .joblib files)
analysis/         One-off analysis, visualization, data prep scripts
tests/            Automated tests (42 tests across 7 files)
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
- **Frontend:** React + TypeScript + Vite + Tailwind v4 + shadcn/ui + framer-motion (built — UC1, UC2, Login, Register)
- **Auth:** Firebase email/password (implemented — frontend AuthContext + backend token verification)
- **Data:** pandas, numpy, joblib

## Key Design Rules

1. **`src/` is the single source of truth** for optimization and prediction logic. The backend imports from `src/`. Never duplicate optimizer code. The `cli/` folder is just an early prototype — all new work goes through the backend API + React frontend.
2. **The deliverable is a web app.** The report specifies React frontend + FastAPI backend + Firebase auth. The CLI was a mockup to prove the algorithm works. The final product must be the web system.
3. **ILP, not greedy.** The backend uses the ILP optimizer from `src/team_builder.py` via a thin wrapper in `backend/app/services/optimizer.py`.
4. **Explanations are the #1 differentiator.** The report positions transparency/explainability as the core competitive advantage. Every recommendation must include human-readable explanations for why players were selected.
5. **Position-specific models.** There are 4 separate ML models (GK, DEF, MID, FWD). Use `position_model_*.joblib` (basic RandomForest) — the advanced models have broken DEF/MID files (4.8KB each).
6. **FPL constraints are strict:** budget cap (default 100.0), max 3 players per club, position requirements (2GK/5DEF/5MID/3FWD for full squad), valid formations for starting XI.

## Two Core Use Cases

### UC1 — Build Initial Squad (15 players)
- User sets budget, formation, optional locked/banned players via the **web UI**
- System validates constraints → predicts season points → ILP optimizer → 15-player squad (11 starters + 4 bench)
- Output includes player list, total cost, expected points, and **short explanations for key selections**
- Exception flows: infeasible constraints → explain conflict; missing data → inform user; model unavailable → fallback; optimization timeout → return best-so-far
- **Status:** COMPLETE end-to-end. ILP optimizer builds 15-player squad. Endpoint returns starters, bench, explanations. React UI live with FIFA-style cards, team kits, glassmorphism, Firebase-auth-gated.

### UC2 — Recommend Weekly Lineup (starting 11 from existing squad)
- User provides/loads their current 15-player squad via the **web UI**
- System validates squad → predicts GW points per player → selects optimal starting XI + bench order
- Output includes starting 11, formation, bench order, expected points, and **short explanations**
- Exception flows: invalid squad → highlight issues; missing GW data → offer refresh; model unavailable → fallback
- **Status:** COMPLETE end-to-end. FPL API fetcher, gameweek predictor (V8 CatBoost models in `Weekly Model/production/`), lineup optimizer (ILP), endpoint all implemented. React UI live with formation override, gameweek picker, captain/VC badges, kit images.

## Database (PostgreSQL via Docker)

Postgres runs in Docker. The `automatic_champion` database is created automatically by the container.

```bash
docker compose up -d        # Start Postgres (runs on localhost:5432)
docker compose down          # Stop Postgres (data persists in pgdata volume)
```

**Connection:** `DATABASE_URL` env var (default: `postgresql+psycopg2://postgres:postgres@localhost:5432/automatic_champion`)

**6 tables:** User, Constraint_Set, Team, Player, Season_Team, Season_Team_Player.
Tables are created via `Base.metadata.create_all()`. Seeded with 20 teams and 784 players (2024-25 season).

**Status:** Schema matches the report ERD. Tables created and seeded. But Constraint_Set.value_json is stored and never read, and Season_Team/Season_Team_Player are written but never queried back.

## Running Things

```bash
# Start database
docker compose up -d

# Activate venv
source .venv/bin/activate

# Create tables (first time only)
python -c "import sys; sys.path.insert(0,'backend'); from app.database import engine, Base; from app.models import *; Base.metadata.create_all(bind=engine)"

# Seed database (first time only)
python backend/scripts/seed_db.py

# Backend server
uvicorn backend.app.main:app --reload --port 8000

# Frontend (once built)
cd frontend && npm run dev

# Tests
python -m pytest tests/ -v

# Train models
python -m training.train_advanced_models
python -m training.train_position_models

# CLI backend tester (interactive — tests UC1 + UC2 against running backend)
python cli/test_backend.py

# CLI prototype (reference only — not the deliverable)
python -m cli.build_team --budget 100 --formation 4-3-3
```

## Test Plan (from report)

| ID | Description | Status |
|----|-------------|--------|
| TC-01 | Build squad with defaults → valid 15 players | Covered (test_full_squad.py) |
| TC-02 | Build squad with user constraints (locked/banned) | Missing |
| TC-03 | Infeasible constraints → explain conflict | Missing |
| TC-04 | Weekly lineup from valid squad → valid 11 + bench | Covered (test_lineup_optimizer.py, test_lineup_endpoint.py) |
| TC-05 | Invalid squad input → block and explain | Partial (validation tests exist) |
| TC-06 | Missing/stale data → warn and offer options | Partial (FPL API fallback tested) |
| NFR-01 | 30 requests, average ≤ 20 seconds | Missing (likely passes) |
| NFR-02 | 5 new users, ≥ 80% success without help | Missing (needs user study) |
| NFR-03 | Swap model with dummy → still works | Missing |

---

## Implementation Plan (Phases)

### Phase 1: Foundation Fixes (est. ~8 dev-days)

**Goal:** Unify architecture, fix inconsistencies, make the backend production-ready.

- [x] **1.1 Create `requirements.txt`** — done: all deps pinned
- [x] **1.2 Replace backend greedy optimizer with ILP** — done: `optimizer.py` wraps `src.team_builder.build_full_squad()`
- [x] **1.3 Align model files** — done: standardized on `position_model_*.joblib`
- [x] **1.4 Create explanation service** — done: `src/explainer.py` with feature importance + human-readable text
- [x] **1.5 Simplify API contract** — done: backend reads from CSV, request takes `{budget, formation, locked_ids, banned_ids}`
- [x] **1.6 Add bench builder to `src/`** — done: `build_full_squad()` in `src/team_builder.py` (single ILP solve for 15 players)
- [x] **1.7 Move DB creds to env var** — done: `database.py` reads `DATABASE_URL` from env
- [x] **1.8 Add solver timeout** — done: `solver.SetTimeLimit(30_000)` in `_build_solver()`

### Phase 2: UC2 — Weekly Lineup (est. ~10 dev-days)

**Goal:** Implement the entire missing UC2 flow.

- [x] **2.1 FPL API data fetcher** — done: `src/fpl_api.py` with caching, bootstrap-static + fixtures
- [x] **2.2 Gameweek prediction model** — done: `src/gameweek_predictor.py` (placeholder-v1, ep_next → ppg → pred/38 fallback). Partner building real model.
- [x] **2.3 Lineup optimizer** — done: `src/lineup_optimizer.py` ILP picks best 11, captain/VC, bench order
- [ ] **2.4 Transfer recommender** — not started (descoped for now)
- [x] **2.5 UC2 API endpoint** — done: `POST /lineup/recommend` in `backend/app/routers/lineup.py`

### Phase 3: React Frontend (est. ~14 dev-days)

**Goal:** Build the web app from the report. Two pages matching the two use cases.

- [x] **3.1 Scaffold** — Vite + React + TypeScript in `/frontend/`
- [x] **3.2 CORS** — middleware in `backend/app/main.py`
- [x] **3.3 API client** — `/frontend/src/api/` typed wrappers
- [x] **3.4 Squad Builder page (UC1)** — budget slider, formation dropdown, locked/banned search, pitch view, explanation panel
- [x] **3.5 Lineup Advisor page (UC2)** — squad input, formation/gameweek override, captain/VC, explanations
- [x] **3.6 Shared components** — AppHeader, ExplanationPanel, PlayerNode (FIFA-style with kits), PitchView, BenchCard
- [x] **3.7 Demo polish** — premium soccer-themed UI: stadium background, glassmorphism, team kits, framer-motion animations, dark mode default

### Phase 4: Auth & Security (est. ~4 dev-days)

- [x] **4.1 Firebase project setup** — email/password enabled
- [x] **4.2 Frontend auth** — `AuthContext`, `Login`/`Register`/`ForgotPassword` pages (split-screen "pre-match warmup" design), token in API headers, `ProtectedRoute`, logout in header
- [x] **4.3 Backend token verification** — `backend/app/auth.py` with `get_current_user` FastAPI dependency, protects `/squad/generate` and `/lineup/recommend`. Service account JSON lives at `~/firebase-admin-automatic-champion.json` (outside repo), backend reads it via `GOOGLE_APPLICATION_CREDENTIALS` env var.

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
- [x] **6.2 Docker Compose** — done: Postgres via `docker-compose.yml`. Backend + frontend containers can be added later.
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

- `advanced_model_DEF.joblib` and `advanced_model_MID.joblib` are only 4.8KB — likely broken/degenerate (using basic RF models instead)
- `Constraint_Set.value_json` is stored in DB but never read by the optimizer
- `src/gameweek_predictor.py` is placeholder-v1 — partner building real GW prediction model
- Season_Team/Season_Team_Player tables are written but never queried back
