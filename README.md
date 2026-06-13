# Automatic Champion

**Automatic Champion** is a Fantasy Premier League (FPL) squad optimizer. It uses machine-learning models to predict player performance and integer linear programming (ILP) to build the optimal squad under official FPL constraints — then explains *why* each pick was made in plain English. The project is delivered as a full web application with Firebase-authenticated accounts that persist named squads.

The core idea is a three-step pipeline: **predict** (ML models score every player), **optimize** (an OR-Tools ILP solver chooses the best legal combination), and **explain** (every recommendation ships with human-readable reasons). The explanation layer is the project's primary differentiator — seasonal picks are explained with RandomForest feature importance, and weekly picks are explained with per-player SHAP values drawn from the model that actually made the prediction.

> Afeka College B.Sc. Software Engineering final project. Authors: **Yuval Davidovits** and **Yuval Garzon**. Supervisor: **Yair Even Zohar**.

---

## Features

- **UC1 — Build Initial Squad (15 players).** Set a budget, a formation, and optional locked/banned players. The system predicts each player's season points, runs the ILP optimizer, and returns a complete 15-player squad (11 starters + 4 bench) with total cost, expected points, and explanations for the key selections.
- **UC2 — Weekly Lineup Advisor (starting 11).** Load a saved squad (or one just generated from UC1), pick a gameweek, and the system predicts per-player gameweek points with the V8 CatBoost models, then selects the optimal starting XI, bench order, captain, and vice-captain.
- **Human-readable explanations (the #1 differentiator).** Seasonal recommendations use RandomForest feature importance; weekly recommendations use per-player SHAP from the V8 CatBoost model, so the explanation always matches the model that produced the prediction.
- **Firebase email/password authentication.** Login, registration, and a forgot-password flow. Protected endpoints verify the Firebase ID token on the backend.
- **Saved squads.** Authenticated users can save up to **10 named squads**, delete them, and switch between them on the Lineup Advisor. Squads persist per user (scoped by Firebase `uid`).

---

## Architecture

Automatic Champion is a layered client–server application. ML models are trained offline and loaded at runtime; all optimization and prediction logic lives in a single shared `src/` core library that both the backend and the CLI prototype import — there is no duplicated business logic.

```
┌──────────────────────────────────────────────────────────┐
│  React + TypeScript frontend (Vite, Tailwind, shadcn/ui)  │
│  Squad Builder · Lineup Advisor · Auth · Saved Squads     │
└───────────────────────────┬──────────────────────────────┘
                            │  HTTPS + Firebase ID token
                            ▼
┌──────────────────────────────────────────────────────────┐
│  FastAPI backend                                          │
│  /squad/generate · /lineup/recommend · /squads · /players │
│  Firebase token verification (backend/app/auth.py)        │
└───────────────────────────┬──────────────────────────────┘
                            │  imports
                            ▼
┌──────────────────────────────────────────────────────────┐
│  src/  — shared core library (single source of truth)     │
│  team_builder (ILP) · gameweek_predictor · lineup_optimizer│
│  explainer (RF) · weekly_explainer (SHAP) · fpl_api        │
└───────────────────────────┬──────────────────────────────┘
              ┌─────────────┴──────────────┐
              ▼                            ▼
┌───────────────────────┐    ┌───────────────────────────────┐
│  PostgreSQL            │    │  Trained models (loaded at     │
│  users · saved squads  │    │  runtime): RF .joblib +        │
│  (SQLAlchemy ORM)      │    │  CatBoost .cbm per position    │
└───────────────────────┘    └───────────────────────────────┘
```

---

## Tech Stack

- **Language:** Python 3.11+
- **ML (seasonal):** scikit-learn RandomForest, one model per position (GK/DEF/MID/FWD)
- **ML (weekly):** CatBoost, one model per position — the V8 model, with built-in SHAP for explanations
- **Optimization:** Google OR-Tools (SCIP/CBC) — Integer Linear Programming
- **Backend:** FastAPI + SQLAlchemy + PostgreSQL + Firebase Admin SDK
- **Frontend:** React + TypeScript + Vite + Tailwind v4 + shadcn/ui + framer-motion
- **Auth:** Firebase email/password (frontend `AuthContext` + backend token verification)
- **Data:** pandas, numpy, joblib

---

## Repository Structure

```
src/             Core library shared by backend + CLI (optimizer, predictor, explainers)
cli/             Interactive CLI tools (prototype / reference — NOT the deliverable)
backend/         FastAPI server + SQLAlchemy ORM + routers + services
frontend/        React + TypeScript + Vite web app (the deliverable UI)
Weekly Model/    V8 weekly CatBoost models, training notebook, and runtime feature table
training/        Training pipelines for the seasonal RandomForest models
analysis/        One-off analysis, visualization, and data-prep scripts
tests/           pytest suite
data/            FPL data — including players_merged_2024-25.csv (the optimizer's player table)
docs/            Seasonal model report, ML report, and supporting charts
models/          Trained .joblib files (RandomForest, one per position)
outputs/         Generated squad CSVs / JSONs
```

---

## Prerequisites

- **Python 3.11+**
- **Node.js** (with npm) for the frontend
- **Docker** (and Docker Compose) to run PostgreSQL
- A **Firebase service-account JSON** for backend token verification. This file lives **outside the repository** and is referenced only via an environment variable — its contents are never committed.

---

## Setup

```bash
# 1. Clone the repository and enter it
git clone <repo-url> Automatic-Champion
cd Automatic-Champion

# 2. Create and activate a virtual environment, then install Python deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Install frontend dependencies
cd frontend && npm install && cd ..

# 4. Start PostgreSQL (runs on localhost:5432; data persists in the pgdata volume)
docker compose up -d

# 5. Point the backend at your Firebase service-account JSON
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/firebase-admin-automatic-champion.json

# 6. Create the database tables (first time, and after model changes)
python -c "import sys; sys.path.insert(0,'backend'); from app.database import engine, Base; from app.models import *; Base.metadata.create_all(bind=engine)"

# 7. Seed the database (first time only — loads teams and players)
python backend/scripts/seed_db.py
```

> Stop the database with `docker compose down` (the `pgdata` volume keeps your data).
> The backend connects via the `DATABASE_URL` env var; the default is
> `postgresql+psycopg2://postgres:postgres@localhost:5432/automatic_champion`,
> which matches the credentials in `docker-compose.yml`.

---

## Running the App

Run the backend and frontend in two separate terminals (both with the virtual environment active for the backend).

**Backend — FastAPI on port 8000:**

```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/firebase-admin-automatic-champion.json \
  uvicorn backend.app.main:app --reload --port 8000
```

**Frontend — Vite dev server on port 5173:**

```bash
cd frontend
npm run dev
```

Then open <http://localhost:5173>. The backend's interactive API docs are available at <http://localhost:8000/docs>.

**Notes:**
- The Firebase service-account JSON lives outside the repo; the backend reads it through `GOOGLE_APPLICATION_CREDENTIALS`. If the variable is unset or the file is missing, the backend still starts and `GET /health` / `GET /players` work, but the authenticated endpoints return `503 Service Unavailable`.
- CORS is preconfigured for `http://localhost:5173` and `http://localhost:3000`.

---

## API Reference

Base URL: `http://localhost:8000`. Authenticated endpoints expect an `Authorization: Bearer <firebase-id-token>` header.

| Method | Path                  | Auth | Description |
|--------|-----------------------|:----:|-------------|
| GET    | `/health`             | No   | Liveness + database connectivity check. |
| GET    | `/players`            | No   | List all selectable players (id, name, position, team, cost). |
| POST   | `/squad/generate`     | Yes  | UC1 — build an optimal 15-player squad. |
| POST   | `/lineup/recommend`   | Yes  | UC2 — recommend the optimal starting XI for a gameweek. |
| GET    | `/squads`             | Yes  | List the current user's saved squads (summaries). |
| POST   | `/squads`             | Yes  | Save a named squad (max 10 per user; names must be unique). |
| GET    | `/squads/{squad_id}`  | Yes  | Fetch one saved squad with its full payload. |
| DELETE | `/squads/{squad_id}`  | Yes  | Delete a saved squad. |

### `POST /squad/generate` (UC1)

**Request** (`SquadGenerateRequest`):

```jsonc
{
  "budget": 100.0,            // 1.0–500.0, default 100.0
  "formation": "4-3-3",       // default "4-3-3"
  "locked_ids": [],           // player ids forced into the squad
  "banned_ids": []            // player ids excluded from the squad
}
```

**Response** (`SquadGenerateResponse`): `formation`, `budget`, `total_cost`, `total_predicted_points`, plus `players` (the 11 starters) and `bench` (4 players). Each player carries `id`, `name`, `team`, `position`, `cost`, `predicted_points`, `is_starter`, `bench_order`, and an `explanations` list. Infeasible constraints return `400` with a descriptive message.

### `POST /lineup/recommend` (UC2)

**Request** (`LineupRecommendRequest`):

```jsonc
{
  "squad": [ /* exactly 15 players: id, name, position, team, cost, pred */ ],
  "gameweek": null,           // null = auto-detect the current gameweek
  "formation": null           // null = auto-pick the best legal formation
}
```

**Response** (`LineupRecommendResponse`): `formation`, `gameweek`, `captain_id`, `vice_captain_id`, `total_gw_points`, `starters` (with captain/vice flags and SHAP explanations), `bench` (ordered), `predictor_version`, and a `warnings` list that surfaces model-unavailable or stale-data fallbacks.

### Saved squads

`POST /squads` takes a `name` (1–50 chars) plus a full `SquadGenerateResponse` payload. The server enforces a 10-squad cap and rejects duplicate names (case-insensitive). `GET /squads` returns lightweight summaries; `GET /squads/{id}` returns the full saved payload. Squads are always scoped to the authenticated user — another user's id returns `404`.

---

## Machine Learning Models

There are two model families, each with one model per position (GK, DEF, MID, FWD), because the way each position earns FPL points is fundamentally different.

### Seasonal models (UC1) — RandomForest

Four scikit-learn RandomForest regressors predict a player's **season** total points. They are trained offline (`python -m training.train_position_models`), stored as `.joblib` files in `models/`, and loaded at runtime by the optimizer. Their feature-importance output drives the UC1 explanations. See `docs/seasonal_model_report.md` for a non-technical writeup of the data, features, and evaluation.

### Weekly models (UC2) — V8 CatBoost

Four CatBoost regressors predict a player's points for the **next gameweek**, tuned with Optuna (100 trials, 5-fold TimeSeriesSplit). They live in `Weekly Model/production/models/` as `model_{GK,DEF,MID,FWD}.cbm`. CatBoost was chosen for native categorical handling (team/opponent strength), native NaN handling (xG features only exist from 2022-23), and the best validation MAE across all four positions. The models expose built-in SHAP values, which power the per-player UC2 explanations.

Held-out test performance (2024-25 season, never seen during training):

| Position    | Test MAE |
|-------------|----------|
| GK          | 0.73     |
| DEF         | 1.03     |
| MID         | 1.00     |
| FWD         | 1.15     |
| **Average** | **0.98** |

Accuracy: 70.5% of predictions land within ±1 point, 85.2% within ±2, and 91.8% within ±3.

> **Runtime dependency:** `Weekly Model/production/test.csv` is a tracked feature table loaded at inference time and is required for UC2 to run. The 130 MB `train.csv` and the raw `Base Data/` folder are gitignored and only needed to retrain the models. Full methodology and experiment history (V1–V10) are in `Weekly Model/README.md`.

---

## Testing

The project uses **pytest**, with roughly **116 tests** across the suite.

```bash
python -m pytest tests/ -v
```

The suite maps to the engineering report's test plan — squad building and constraints (TC-01/TC-02), infeasible-constraint handling (TC-03), weekly lineup generation (TC-04), invalid-input handling (TC-05), missing/stale-data fallbacks (TC-06), and the model-swap robustness check (NFR-03, `tests/test_model_swap.py`) — alongside auth, saved-squads CRUD, the FPL API wrapper, and both explanation services.

---

## Project Context

Automatic Champion is the final project for the **B.Sc. in Software Engineering at Afeka College of Engineering**.

- **Authors:** Yuval Davidovits, Yuval Garzon
- **Supervisor:** Yair Even Zohar

The deliverable is the web application (React frontend + FastAPI backend + Firebase auth). The CLI tools under `cli/` are a prototype kept for reference and are not part of the deliverable.

### Future work

- **Transfer recommender** — suggest weekly transfers in/out of an existing squad (descoped from the current deliverable).
- Further weekly-model experiments beyond V8, and a richer constraint-persistence story.
