# Backend — Automatic Champion API

FastAPI server for the Automatic Champion FPL squad optimizer.

## Prerequisites

- Python 3.11+
- PostgreSQL (via Docker)
- Virtual environment with dependencies installed

## Setup

```bash
# From the project root

# 1. Start PostgreSQL
docker compose up -d

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create database tables (first time only)
python -c "import sys; sys.path.insert(0,'backend'); from app.database import engine, Base; from app.models import *; Base.metadata.create_all(bind=engine)"

# 5. Seed database with player data (first time only)
python backend/scripts/seed_db.py
```

## Running the Server

```bash
uvicorn backend.app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## API Endpoints

### `GET /health`
Health check. Returns `{"status": "ok"}` if the server and database are running.

### `POST /squad/generate` (UC1 — Build Squad)
Build an optimal 15-player FPL squad.

**Request:**
```json
{
  "budget": 100.0,
  "formation": "4-3-3",
  "locked_ids": [],
  "banned_ids": []
}
```

All fields have defaults — an empty `{}` body works.

**Response:**
```json
{
  "formation": "4-3-3",
  "budget": 100.0,
  "total_cost": 98.5,
  "total_predicted_points": 1493.6,
  "players": [
    {
      "id": "2",
      "name": "Alisson Ramses Becker",
      "team": "Liverpool",
      "position": "GK",
      "cost": 5.5,
      "predicted_points": 145.6,
      "is_starter": true,
      "bench_order": null,
      "explanations": [
        {
          "feature": "price_now",
          "value": 55,
          "importance": 0.35,
          "explanation": "Cost: a good fit for the budget while keeping quality high."
        }
      ]
    }
  ],
  "bench": [ ... ]
}
```

### `POST /lineup/recommend` (UC2 — Weekly Lineup)
Recommend the optimal starting 11 from a 15-player squad for a specific gameweek.

**Request:**
```json
{
  "squad": [
    {"id": "2", "name": "Alisson", "position": "GK", "team": "Liverpool", "cost": 5.5, "pred": 145.6},
    ... (15 players total)
  ],
  "gameweek": null,
  "formation": null
}
```

- `squad`: exactly 15 players (2 GK, 5 DEF, 5 MID, 3 FWD)
- `gameweek`: optional — auto-detects current gameweek if null
- `formation`: optional — tries all 7 formations and picks the best if null

**Response:**
```json
{
  "formation": "4-3-3",
  "gameweek": 31,
  "captain_id": "2",
  "vice_captain_id": "339",
  "total_gw_points": 72.5,
  "starters": [
    {
      "id": "2",
      "name": "Alisson",
      "position": "GK",
      "team": "Liverpool",
      "gw_points": 6.1,
      "is_captain": true,
      "is_vice_captain": false,
      "explanations": [...]
    }
  ],
  "bench": [
    {
      "id": "456",
      "name": "...",
      "position": "DEF",
      "team": "...",
      "gw_points": 3.2,
      "bench_order": 1,
      "explanations": [...]
    }
  ]
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+psycopg2://postgres:postgres@localhost:5432/automatic_champion` | PostgreSQL connection string |
