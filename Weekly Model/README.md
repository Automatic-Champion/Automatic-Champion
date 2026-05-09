# Weekly GW Points Prediction — Production Model

Predicts how many FPL points a player will score **next gameweek**, given their current stats and recent history. Used in UC2 (Weekly Lineup Advisor) to pick the optimal starting XI from a manager's squad.

---

## Folder Structure

```
weekly_model_data/
├── production/
│   ├── models/
│   │   ├── model_GK.cbm
│   │   ├── model_DEF.cbm
│   │   ├── model_MID.cbm
│   │   └── model_FWD.cbm
│   ├── weeklyModels_Production.ipynb   ← training notebook
│   ├── build_data.py                   ← data pipeline
│   ├── train.csv                       ← 2016-17 → 2023-24 (190,825 rows)
│   ├── test.csv                        ← 2024-25 (26,499 rows)
│   └── feature_importance.csv
├── Base Data/                          ← raw FPL season data
└── Archive/                            ← all previous versions (V7–V10)
```

---

## Results

Four separate models, one per position. Test set: **2024-25 season** (never seen during training).

| Position | Test MAE | Val MAE |
|----------|----------|---------|
| GK       | 0.7295   | 0.6125  |
| DEF      | 1.0278   | 1.0551  |
| MID      | 0.9998   | 0.9394  |
| FWD      | 1.1534   | 1.0066  |
| **Average** | **0.9776** | |

| Threshold | Accuracy |
|-----------|----------|
| Within ±1 pt  | **70.5%** |
| Within ±2 pts | **85.2%** |
| Within ±3 pts | **91.8%** |

**Real examples from 2024-25:**

| Player | GW | Real | Predicted |
|--------|-----|------|-----------|
| Mohamed Salah | 22 | 8 pts | 6.0 pts |
| Leandro Trossard | 33 | 8 pts | 6.3 pts |
| Altay Bayindir (bench) | 18 | 0 pts | 0.0 pts |
| Cole Palmer (hat-trick) | 5 | 25 pts | 4.1 pts |
| Erling Haaland (blank) | 25 | 0 pts | 5.5 pts |

The last two rows show the irreducible error — surprise events no historical model can predict.

---

## Model

**CatBoost per position**, tuned with Optuna (100 trials, 5-fold TimeSeriesSplit).

Chosen over HistGBR / XGBoost / LightGBM for three reasons:
- Native categorical handling — learns team/opponent strength without manual encoding
- Native NaN handling — xG features are only available from 2022-23, no imputation needed
- Best validation MAE across all four positions

**Features:** current GW stats (minutes, ICT, BPS, xG, team goals, value, transfers) + rolling averages over 1/3/5/7 GWs. Position-specific extras: saves/clean sheets for GK, goals/assists for FWD/MID.

**Split:**
```
Train (Optuna CV): 2016-17 → 2022-23
Val:               2023-24
Test:              2024-25  ← reported numbers above
```

---

## How to Use

```python
from catboost import CatBoostRegressor, Pool

model = CatBoostRegressor()
model.load_model('weekly_model_data/production/models/model_MID.cbm')

# team and opponent_team must be strings
pred = model.predict(Pool(X, cat_features=['team', 'opponent_team']))
```

---

## How to Reproduce

```bash
cd weekly_model_data/production
source ../../.venv/bin/activate

# Rebuild data
python build_data.py

# Retrain — open in Jupyter Lab with "Python (Automatic Champion)" kernel
jupyter lab weeklyModels_Production.ipynb
```

---

## Experiment History

| Version | Approach | Avg MAE | Outcome |
|---------|----------|---------|---------|
| V1–V6 | Early prototypes | — | Superseded by V7 pipeline rewrite |
| V7 | HistGradientBoosting + OrdinalEncoder | ~1.00 | Baseline |
| V7B | HistGBR on xG seasons only (2022-24) | Worse | Less data hurt more than richer features helped |
| V7D | Ensemble of V7 + V8 | Worse | Averaging added noise, not signal |
| **V8** | **CatBoost, all seasons** | **0.9776** | **Production** |
| V9 | Two-stage: classifier + regressor | 0.9766 ≈ V8 | CatBoost already handles this implicitly |
| V10 | V8 + opponent rolling form features | 0.9781 ≈ V8 | CatBoost extracts this from opponent_team categorical |

---

## Authors

Yuval Davidovits, Yuval Garzon — Afeka College B.Sc. Software Engineering, 2026
