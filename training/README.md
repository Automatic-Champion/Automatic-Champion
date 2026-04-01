# Training — Evaluation Framework

Standardized evaluation framework for measuring ML model quality against fixed data splits and a recorded baseline.

## Quick Start

```bash
# Generate (or regenerate) baseline metrics from the current models
python -m training.run_baseline
# or: python training/run_baseline.py
```

This produces `training/baseline_metrics.csv` — the fixed baseline all future models must beat.

## Data Splits

| Split   | Seasons                          | Purpose                                     |
|---------|----------------------------------|---------------------------------------------|
| Train   | 2019-20, 2020-21, 2021-22       | Model training (3 seasons)                  |
| Test    | 2022-23                         | Hyperparameter tuning & CV (1 season)       |
| Unseen  | 2023-24                         | Final reporting only — never optimize against this |

Load them with:

```python
from training.eval_framework import load_train_test_unseen

train_df, test_df, unseen_df = load_train_test_unseen()
```

## Using the Framework in a New Training Script

```python
from training.eval_framework import (
    load_train_test_unseen,
    evaluate_all_positions,
    compare_to_baseline,
)

# 1. Load canonical splits
train_df, test_df, unseen_df = load_train_test_unseen()

# 2. Build your features and train models (one per position)
#    models = {1: gk_model, 2: def_model, 3: mid_model, 4: fwd_model}
feature_cols = [...]  # your feature columns
models = {}
for pos in [1, 2, 3, 4]:
    # ... train models[pos] on train_df ...
    pass

# 3. Evaluate on the test set
results_df = evaluate_all_positions(models, feature_cols, test_df)

# 4. Compare to baseline
comparison_df = compare_to_baseline(results_df)
```

## Baseline Metrics

Current baseline (position_model_*.joblib — RandomForest, trained on 2019-2023):

### Test Set (2022-23)

| Position | MAE   | RMSE  | R²    | Median AE | N   |
|----------|-------|-------|-------|-----------|-----|
| GK       |  7.02 | 12.09 | 0.946 |      1.63 |  85 |
| DEF      |  9.41 | 13.08 | 0.898 |      7.54 | 268 |
| MID      | 11.09 | 16.26 | 0.896 |      7.40 | 338 |
| FWD      | 11.45 | 17.65 | 0.901 |      8.04 |  93 |

### Unseen Set (2023-24)

| Position | MAE   | RMSE  | R²    | Median AE | N   |
|----------|-------|-------|-------|-----------|-----|
| GK       | 14.35 | 28.04 | 0.547 |      0.93 | 100 |
| DEF      | 24.38 | 29.46 | 0.401 |     22.55 | 285 |
| MID      | 25.06 | 36.56 | 0.486 |     16.03 | 374 |
| FWD      | 27.69 | 44.79 | 0.310 |     13.83 | 113 |

The large gap between test and unseen performance suggests the models overfit to the training window or that 2023-24 had significantly different patterns.

## API Reference

- **`load_train_test_unseen()`** — Returns `(train_df, test_df, unseen_df)` with fixed season splits.
- **`evaluate_model(model, X_test, y_test, position_name)`** — Returns metrics dict for a single model.
- **`evaluate_all_positions(models, feature_cols, test_df)`** — Evaluates all 4 positions, returns summary DataFrame.
- **`compare_to_baseline(results_df, baseline_csv)`** — Compares results against the saved baseline CSV.
