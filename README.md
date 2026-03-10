# firstdataset

QSAR biodegradation dataset curation and baseline modeling code.

## Setup

```sh
source "/Users/mannz/Desktop/polymer degredation/firstdataset/activate-project.sh"
```

## Main files

- `src/firstdataset/data.py`: dataset loading and curation
- `src/firstdataset/modeling.py`: baseline classification and regression models
- `scripts/run_regression_baselines.py`: generic regression runner for any CSV with a numeric target

## Run regression baselines

```sh
python "/Users/mannz/Desktop/polymer degredation/firstdataset/scripts/run_regression_baselines.py" \
  --csv "/path/to/your_regression_dataset.csv" \
  --target your_numeric_target_column \
  --output "/Users/mannz/Desktop/polymer degredation/firstdataset/regression_scores.json"
```
