# CLAUDE.md — Automatic Champion

## What This Project Is

Automatic Champion is a Fantasy Premier League (FPL) squad optimizer — a B.Sc. Software Engineering final project (Afeka College). It uses ML models to predict player performance and integer linear programming (ILP) to build optimal squads under FPL constraints. Delivered as a web app with Firebase-authenticated accounts that persist named squads.

**Authors:** Yuval Davidovits, Yuval Garzon
**Deadline:** July 17–20, 2026 (final hand-in + defense)
**Engineering report:** Submitted January 2026

---

## Repo Structure

```
src/                       Core library — shared by backend and CLI (DO NOT duplicate logic)
  team_builder.py             ILP optimizer (OR-Tools), prediction, constraint validation, full squad builder
  explainer.py                Seasonal explanation service (RandomForest feature importance + human-readable text)
  weekly_explainer.py         Weekly explanation service (per-player SHAP over the V8 features the predictor used)
  fpl_api.py                  FPL API wrapper (bootstrap-static, fixtures, caching)
  gameweek_predictor.py       GW points predictor — loads V8 CatBoost models, lazy fallback to ep_next/ppg
  lineup_optimizer.py         Lineup optimizer — ILP picks best 11 from 15, captain/VC, bench order
cli/                       CLI tools (reference / prototype only — NOT the deliverable)
  build_team.py               Interactive menu-driven team builder
  test_backend.py             Interactive CLI to test backend API (UC1 + UC2)
backend/                   FastAPI server + PostgreSQL
  app/
    main.py                   FastAPI app, /health, /players, router registration, lifespan-init for Firebase
    auth.py                   Firebase Admin SDK init + get_current_user FastAPI dependency
    database.py               SQLAlchemy engine (PostgreSQL)
    deps.py                   Dependency injection (DB session)
    models.py                 ORM models (7 tables)
    schemas.py                Pydantic request/response schemas
    routers/
      squad.py                POST /squad/generate (UC1, auth-protected)
      lineup.py               POST /lineup/recommend (UC2, auth-protected)
      saved_squads.py         GET/POST/DELETE /squads (per-user persistence, auth-protected)
    services/
      optimizer.py            Thin wrapper around src.team_builder.build_full_squad()
  scripts/
    seed_db.py                Seed DB from FPL GitHub CSV data
frontend/                  React + TypeScript + Vite + Tailwind v4 + shadcn/ui + framer-motion
  src/
    api/                       Typed API client + types
    components/                Shared UI: pitch, squad, auth hero, form, common
    context/                   AuthContext, SquadContext, ThemeContext
    layout/                    AppHeader (user email, logout, active squad badge), sidebar
    lib/                       firebase.ts, teamKits.ts (team-name → kit PNG), explanationCategories.tsx (badge colours/icons shared by ExplanationPanel + Stats Guide), utils
    pages/                     SquadBuilder, LineupAdvisor, StatsGuide, Login, Register, ForgotPassword
  public/
    kits/                      Premier League team kit PNGs (17 teams × 3 variants = 51 files)
Weekly Model/              V8 weekly prediction model
  production/
    models/                    model_{GK,DEF,MID,FWD}.cbm (CatBoost per position)
    test.csv                   Feature table loaded at inference (21MB, tracked in git)
    build_data.py              Regenerates train/test from Base Data (Base Data is NOT in git)
    weeklyModels_Production.ipynb  Training notebook (Optuna 100-trial, 5-fold TimeSeriesSplit)
  README.md                   V8 docs: data, training methodology, MAE table, experiment history V1–V10
docs/
  seasonal_model_report.md    Non-ML-audience writeup of the seasonal RF models
  seasonal_model_images/      10 PNG charts (actual-vs-predicted, residuals, feature importance, etc.)
training/                  ML training pipelines for seasonal RF models (run manually, produce .joblib)
analysis/                  One-off analysis, visualization, data prep scripts
tests/                     Automated test suite (~116 tests across multiple files)
data/
  base/                       10 seasons of raw FPL data (2016-17 through 2025-26)
  historical_exports/         Pre-built training CSVs per season
  season_comparisons/         Season delta CSVs
  players_merged_2024-25.csv  Current season player table the optimizer reads
models/                    Trained .joblib files (4 basic RF per position)
outputs/                   Generated squad CSVs/JSONs
visuals/                   Generated charts and metrics CSVs
```

## Tech Stack

- **Language:** Python 3.11+
- **ML (seasonal):** scikit-learn RandomForest per position (GK/DEF/MID/FWD)
- **ML (weekly):** CatBoost per position — V8 model with built-in SHAP for explanations
- **Optimization:** Google OR-Tools (SCIP/CBC solver) — Integer Linear Programming
- **Backend:** FastAPI + SQLAlchemy + PostgreSQL + Firebase Admin SDK
- **Frontend:** React + TypeScript + Vite + Tailwind v4 + shadcn/ui + framer-motion
- **Auth:** Firebase email/password (frontend AuthContext + backend token verification + ForgotPassword flow)
- **Data:** pandas, numpy, joblib

## Key Design Rules

1. **`src/` is the single source of truth** for optimization and prediction logic. Both the backend and the CLI prototype import from `src/`. Never duplicate optimizer or predictor code.
2. **The deliverable is a web app.** React frontend + FastAPI backend + Firebase auth. The CLI is a prototype, not the final product.
3. **ILP, not greedy.** The backend uses the ILP optimizer from `src/team_builder.py` via a thin wrapper in `backend/app/services/optimizer.py`.
4. **Explanations are the #1 differentiator.** Every recommendation includes human-readable reasons. Seasonal recommendations use RandomForest feature importance; weekly recommendations use per-player SHAP from the V8 CatBoost model, so the explanation matches the model that actually made the prediction.
5. **Position-specific models.** Four separate models per use case (seasonal RF, weekly CatBoost), one per position.
6. **FPL constraints are strict:** budget cap (default 100.0), max 3 players per club, position requirements (2GK/5DEF/5MID/3FWD for full squad), valid formations for starting XI.

## Two Core Use Cases

### UC1 — Build Initial Squad (15 players)
- User sets budget, formation, optional locked/banned players via the **web UI**
- System validates constraints → predicts season points → ILP optimizer → 15-player squad (11 starters + 4 bench)
- Output includes player list, total cost, expected points, and **short explanations for key selections**
- Exception flows: infeasible constraints → explain conflict; missing data → inform user; model unavailable → fallback; optimization timeout → return best-so-far
- **Status:** COMPLETE end-to-end. ILP optimizer builds 15-player squad. Endpoint returns starters, bench, explanations. React UI live with FIFA-style cards, team kits, glassmorphism, Firebase-auth-gated.

### UC2 — Recommend Weekly Lineup (starting 11 from existing squad)
- User loads a saved squad (or just-generated one from UC1) via the **web UI**
- System validates squad → predicts GW points per player using **V8 CatBoost** → selects optimal starting XI + bench order
- Output includes starting 11, formation, bench order, expected points, captain/VC, and **per-player SHAP-based explanations**
- Exception flows: invalid squad → highlight issues; missing GW data → offer refresh; model unavailable → fallback to ep_next/ppg chain
- **Status:** COMPLETE end-to-end. V8 CatBoost models (avg MAE 0.98), lineup optimizer (ILP), endpoint all implemented. React UI live with formation override, gameweek picker, captain/VC badges, kit images, SHAP-driven category badges (form, fixture, performance, attacking, defensive, reliability, value, trending).

## Account Features

- **Firebase email/password auth** (login, register, forgot password)
- **Saved squads:** authenticated users can save up to **10 named squads**, delete them, and switch between them on the Lineup Advisor (squads persist in `saved_squads` table, scoped by Firebase `uid`)
- Active-squad badge in the header; "Saved as ..." pill on Squad Builder; cross-session loading via `getSavedSquad(id)` → populates `SquadContext`

## Database (PostgreSQL via Docker)

Postgres runs in Docker. The `automatic_champion` database is created automatically by the container.

```bash
docker compose up -d        # Start Postgres (runs on localhost:5432)
docker compose down          # Stop Postgres (data persists in pgdata volume)
```

**Connection:** `DATABASE_URL` env var (default: `postgresql+psycopg2://postgres:postgres@localhost:5432/automatic_champion`)

**7 tables:** User, Constraint_Set, Team, Player, Season_Team, Season_Team_Player, **SavedSquad** (new — `id`, `user_uid`, `name`, `payload_json`, `created_at`).
Tables are created via `Base.metadata.create_all()`. Seeded with 20 teams and 784 players (2024-25 season).

**Status:** Schema matches the report ERD plus the new SavedSquad table. Saved squads are fully wired (read + write). `Constraint_Set.value_json` is still stored but never read; `Season_Team` / `Season_Team_Player` are still written but never queried back — neither blocks the deliverable.

## Running Things

```bash
# Start database
docker compose up -d

# Activate venv
source .venv/bin/activate

# Create tables (first time + after model changes)
python -c "import sys; sys.path.insert(0,'backend'); from app.database import engine, Base; from app.models import *; Base.metadata.create_all(bind=engine)"

# Seed database (first time only)
python backend/scripts/seed_db.py

# Backend server (needs Firebase credentials env var)
GOOGLE_APPLICATION_CREDENTIALS=/Users/yuvaldavidovits/firebase-admin-automatic-champion.json \
  uvicorn backend.app.main:app --reload --port 8000

# Frontend
cd frontend && npm run dev

# Tests
python -m pytest tests/ -v

# Train seasonal models (regenerate .joblib)
python -m training.train_position_models

# Retrain weekly V8 (run notebook)
jupyter lab "Weekly Model/production/weeklyModels_Production.ipynb"

# CLI backend tester (interactive)
python cli/test_backend.py
```

## Test Plan (from report)

| ID | Description | Status |
|----|-------------|--------|
| TC-01 | Build squad with defaults → valid 15 players | Covered (test_full_squad.py) |
| TC-02 | Build squad with user constraints (locked/banned) | Covered (test_squad_constraints.py) |
| TC-03 | Infeasible constraints → explain conflict | Covered (test_squad_constraints.py) |
| TC-04 | Weekly lineup from valid squad → valid 11 + bench | Covered (test_lineup_optimizer.py, test_lineup_endpoint.py) |
| TC-05 | Invalid squad input → block and explain | Covered (lineup: test_lineup_endpoint.py; squad: test_squad_constraints.py) |
| TC-06 | Missing/stale data → warn and offer options | Covered (lineup response surfaces `warnings` on V8 fallback; test_lineup_endpoint.py + test_gameweek_predictor.py) |
| NFR-01 | 30 requests, average ≤ 20 seconds | Covered (test_performance.py; measured avg 0.87s — UC1 0.67s / UC2 1.07s) |
| NFR-02 | 5 new users, ≥ 80% success without help | Missing (can be informal) |
| NFR-03 | Swap model with dummy → still works | Covered (test_model_swap.py) |

**Test files (latest):**
- `test_full_squad.py` — TC-01 coverage
- `test_lineup_optimizer.py`, `test_lineup_endpoint.py` — TC-04 coverage; endpoint tests patched for Firebase token mocking; lineup endpoint also asserts TC-06 `warnings` (model-unavailable / partial-fallback / healthy)
- `test_gameweek_predictor.py` — V8 inference paths + `get_feature_rows` drift guard + `predict_gameweek_points_with_meta` degradation metadata (TC-06)
- `test_weekly_explainer.py` — SHAP-based weekly explanations (6 cases)
- `test_auth.py` — Firebase auth dependency (9 cases — public/missing/invalid/valid token paths)
- `test_saved_squads.py` — per-user squad CRUD (13 cases — in-memory SQLite via dependency_overrides)
- `test_squad_constraints.py` — TC-02/TC-03/TC-05 squad-build constraints (10 cases — locked/banned happy paths, all infeasibility errors, router 400 mapping)
- `test_model_swap.py` — NFR-03 model-agnosticism (2 cases — every position model swapped for a constant-returning DummyModel, still builds a valid 15)
- `test_performance.py` — NFR-01 performance (1 case — 30 real requests, 15 UC1 + 15 UC2, asserts avg ≤ 20s; measured 0.87s avg)
- Full suite: **116 tests** passing

---

## Implementation Plan (Phases)

### Phase 1: Foundation Fixes (DONE)
- [x] **1.1 Create `requirements.txt`**
- [x] **1.2 Replace backend greedy optimizer with ILP**
- [x] **1.3 Align model files** — standardized on `position_model_*.joblib`
- [x] **1.4 Create explanation service** — `src/explainer.py`
- [x] **1.5 Simplify API contract**
- [x] **1.6 Add bench builder to `src/`**
- [x] **1.7 Move DB creds to env var**
- [x] **1.8 Add solver timeout**

### Phase 2: UC2 — Weekly Lineup (DONE, transfer recommender descoped)
- [x] **2.1 FPL API data fetcher**
- [x] **2.2 Gameweek prediction model** — V8 CatBoost per position landed (Yuval Garzon). MAE: GK 0.73 / DEF 1.03 / MID 1.00 / FWD 1.15 / avg 0.98 on 2024-25 held-out test
- [x] **2.3 Lineup optimizer**
- [ ] **2.4 Transfer recommender** — descoped
- [x] **2.5 UC2 API endpoint**
- [x] **2.6 V8 SHAP-based per-player weekly explanations** — `src/weekly_explainer.py`; explanations now match the model that actually predicted

### Phase 3: React Frontend (DONE)
- [x] **3.1 Scaffold** — Vite + React + TypeScript in `/frontend/`
- [x] **3.2 CORS** — middleware in `backend/app/main.py`
- [x] **3.3 API client** — `/frontend/src/api/` typed wrappers
- [x] **3.4 Squad Builder page (UC1)** — budget slider, formation dropdown, locked/banned search, pitch view, explanation panel, save button
- [x] **3.5 Lineup Advisor page (UC2)** — squad input, formation/gameweek override, captain/VC, explanations, saved-squads loader
- [x] **3.6 Shared components** — AppHeader (with user email + active squad badge + logout), ExplanationPanel (with form/fixture categories), PlayerNode (FIFA-style with kits), PitchView, BenchCard, SaveSquadDialog, SavedSquadsList
- [x] **3.7 Demo polish** — premium soccer-themed UI: stadium background, glassmorphism, team kits, framer-motion animations, dark mode default
- [x] **3.8 Stats Guide page** — third sidebar tab at `/guide`: plain-English reference for the on-screen numbers, the eight explanation badges, and the football stats the reasons cite (xG/xA/ICT/BPS/xGC/…), with client-side search. Static content, no API change

### Phase 4: Auth & Security (DONE)
- [x] **4.1 Firebase project setup** — email/password enabled
- [x] **4.2 Frontend auth** — `AuthContext` (login/register/logout/resetPassword), `Login`/`Register`/`ForgotPassword` pages with split-screen "pre-match warmup" design, token in API headers, `ProtectedRoute`, logout in header
- [x] **4.3 Backend token verification** — `backend/app/auth.py` with `get_current_user` FastAPI dependency, lazy init via lifespan so backend imports cleanly without credentials (for CI / fresh clones). Protects `/squad/generate`, `/lineup/recommend`, all `/squads`. Service account JSON at `~/firebase-admin-automatic-champion.json` (outside repo), backend reads via `GOOGLE_APPLICATION_CREDENTIALS`

### Phase 4.5: Saved Squads (DONE)
- [x] Backend: `SavedSquad` ORM model + `/squads` router (list/create/get/delete) with 10-cap, duplicate-name rejection, silent ownership 404s
- [x] Frontend: `SaveSquadDialog`, `SavedSquadsList`, `currentSquadName` + `setSquadFromSaved` on `SquadContext`, "Active squad: ..." badge in header
- [x] Tests: 13 cases in `test_saved_squads.py`

### Phase 5: Testing (~partial)
- [ ] **5.1 TC-01** — all 7 formations, full 15-player squad, budget edge cases (basic coverage exists, not exhaustive)
- [x] **5.2 TC-02** — locked/banned constraints (test_squad_constraints.py)
- [x] **5.3 TC-03** — infeasible constraints → descriptive error (test_squad_constraints.py)
- [x] **5.4 TC-04** — weekly lineup from valid squad
- [x] **5.5 TC-05** — invalid input → block and explain (lineup + squad endpoints)
- [x] **5.6 TC-06** — missing/stale data → warn (lineup response `warnings` field on V8 fallback)
- [x] **5.7 NFR-01** — 30 requests, avg ≤ 20 seconds (test_performance.py; measured 0.87s avg)
- [x] **5.8 NFR-03** — swap model with DummyModel → system still works (test_model_swap.py)
- [x] **5.9 API integration tests** — auth, saved squads, weekly explainer all covered

### Phase 6: Polish & Documentation
- [ ] **6.1 README** — setup, architecture, how to run, API docs (NOT YET WRITTEN — needed for grading)
- [x] **6.2 Docker Compose** — Postgres via `docker-compose.yml`. Backend + frontend containers can be added later.
- [ ] **6.3 Backend logging** — Python logging module
- [ ] **6.4 Clean up dead code** — remove greedy optimizer, unused scripts
- [ ] **6.5 Wire Constraint_Set.value_json** — likely descopable; saved squads now cover the per-user persistence story

---

## Parallelization (2 students)

```
Student A (backend-focused):  Phase 1 → Phase 2 → Phase 4 backend → Phase 5 backend tests
Student B (frontend-focused): Phase 3 scaffold → Phase 3 UI → Phase 4 frontend → Phase 5 frontend tests
Both:                         Phase 6
```

## What Can Be Descoped If Time Is Short

- Transfer recommender (Task 2.4) — already descoped
- NFR-02 usability study — do informally with classmates
- Wire `Constraint_Set.value_json` (Task 6.5) — saved squads cover the persistence story
- Robust optimization / Monte Carlo — report mentions as enhancement, explicitly optional
- ML improvements beyond V8 (V11 / Experiment A) — V8 already strong; further tuning is enhancement

## What Cannot Be Cut

- UC1 end-to-end via web UI with explanations
- UC2 end-to-end via web UI with explanations
- Firebase auth (or equivalent)
- Tests matching TC-01 through TC-06 and NFR-01/NFR-03
- Explanation system in the API response

## Known Issues

- `advanced_model_DEF.joblib` and `advanced_model_MID.joblib` are only 4.8KB — likely broken/degenerate (basic RF models used instead)
- `Constraint_Set.value_json` is stored in DB but never read by the optimizer (saved squads supersede this need)
- `Season_Team` / `Season_Team_Player` tables are written but never queried back
- `Weekly Model/Base Data/` and `train.csv` (130MB) are gitignored — only `test.csv` (21MB, runtime requirement) ships through git. To retrain V8, the Base Data folder must be obtained separately
- V11 weekly-model experiment failed (+0.65% MAE vs V8) — files remain untracked under `Weekly Model/production_v11/`. Next attempt would be Experiment A (`opponent_team_season` categorical + slim feature set), not started

## Remaining Work for the July 17–20 Defense

The report's test plan is now complete: TC-01–TC-06, NFR-01, and NFR-03 are all Covered (test_full_squad.py, test_squad_constraints.py, test_lineup_*.py, test_gameweek_predictor.py, test_model_swap.py, test_performance.py). NFR-02 is a manual usability study (not automatable). The README (Phase 6.1) is written. Remaining, in priority order:

1. **NFR-02 informal user study** — five classmates, ≥80% success without help (manual; not automatable).
2. **Backend logging** (Phase 6.3) and **dead code cleanup** (Phase 6.4).
3. (Optional) Frontend: surface the TC-06 lineup `warnings` as a UI banner on the Lineup Advisor (option C — backend already returns the field).
4. (Optional) ML enhancements — Experiment A for the weekly model, transfer recommender.
