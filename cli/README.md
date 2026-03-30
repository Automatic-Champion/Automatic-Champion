# CLI Backend Tester

Interactive CLI that exercises the Automatic Champion backend API. Tests both UC1 (squad generation) and UC2 (lineup recommendation) before the React frontend is built.

## Prerequisites

- Python 3.11+ with the project virtualenv activated
- Backend running (in a separate terminal):

```bash
source .venv/bin/activate
uvicorn backend.app.main:app --reload --port 8000
```

## Usage

```bash
python cli/test_backend.py              # default: http://localhost:8000
python cli/test_backend.py --url http://localhost:5000  # custom URL
```

## Menu Options

| Option | Description |
|--------|-------------|
| **[1] Build Squad** | Use the optimizer to build an optimal 15-player squad. Set budget, formation, and optionally lock/ban players by searching their names. Shows starters, bench, and explanations for each pick. |
| **[2] Enter My Squad** | Manually build your squad by searching and picking players one at a time. Shows progress (how many of each position picked). Type `auto` to let the optimizer fill remaining slots. Type `undo` to remove the last pick. |
| **[3] Recommend Lineup** | Get the best starting 11 from your current squad for a specific gameweek. Shows captain, vice-captain, and bench order. Requires a squad from option 1, 2, or 6. |
| **[4] Browse Players** | Search all available players by name, team, or position. |
| **[5] View Current Squad** | Display the squad you're currently working with. |
| **[6] Saved Squads** | List and load previously saved squads. |
| **[7] Quick Test** | Runs build squad + lineup recommendation with all defaults. Good for a quick smoke test. |

## Typical Workflows

- **"I want the optimizer to pick the best team"** — Option 1, then Option 3
- **"I want to enter my actual FPL team and get lineup advice"** — Option 2, then Option 3
- **"I just want to check the backend works"** — Option 7
