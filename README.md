# firstdataset

QSAR biodegradation dataset curation, modeling, and validation code.

## Setup

```sh
source "/Users/mannz/Desktop/polymer degredation/firstdataset/activate-project.sh"
```

## Main files

- `src/firstdataset/data.py`: dataset loading and curation
- `src/firstdataset/modeling.py`: baseline classification and regression models
- `src/firstdataset/week8_validation.py`: cross-environment validation helpers
- `src/firstdataset/week9_validation.py`: 5-fold stratified validation with SMOTE support
- `scripts/run_regression_baselines.py`: generic regression runner for any CSV with a numeric target
- `scripts/train_week6_fnn.py`: Week 6 feedforward neural network run
- `scripts/train_week7_gnn.py`: Week 7 descriptor-graph prototype run
- `scripts/run_week8_cross_environment.py`: Week 8 proxy environment validation
- `scripts/run_week9_validation.py`: Week 9 validation and imbalance comparison

## Run regression baselines

```sh
python "/Users/mannz/Desktop/polymer degredation/firstdataset/scripts/run_regression_baselines.py" \
  --csv "/path/to/your_regression_dataset.csv" \
  --target your_numeric_target_column \
  --output "/Users/mannz/Desktop/polymer degredation/firstdataset/regression_scores.json"
```

## Run Week 9 validation

```sh
PYTHONPATH="/Users/mannz/Desktop/polymer degredation/firstdataset/src" \
python "/Users/mannz/Desktop/polymer degredation/firstdataset/scripts/run_week9_validation.py"
```

This writes:

- `reports/week9_fold_diagnostics.csv`
- `reports/week9_model_results.csv`
- `reports/week9_validation_metrics.json`
- `WEEK9_VALIDATION_SUMMARY.txt`
