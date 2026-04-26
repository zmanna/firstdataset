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
- `src/firstdataset/week10_features.py`: Week 10 feature engineering tiers and evaluation
- `src/firstdataset/week11_analysis.py`: Week 11 feature importance and selection
- `src/firstdataset/week12_uncertainty_analysis.py`: Week 12 uncertainty, calibration, and reliability analysis
- `scripts/run_regression_baselines.py`: generic regression runner for any CSV with a numeric target
- `scripts/train_week6_fnn.py`: Week 6 feedforward neural network run
- `scripts/train_week7_gnn.py`: Week 7 descriptor-graph prototype run
- `scripts/run_week8_cross_environment.py`: Week 8 proxy environment validation
- `scripts/run_week9_validation.py`: Week 9 validation and imbalance comparison
- `scripts/run_week10_feature_engineering.py`: Week 10 feature engineering comparison
- `scripts/run_week11_feature_analysis.py`: Week 11 feature importance and selection
- `scripts/run_week12_uncertainty_analysis.py`: Week 12 uncertainty and calibration analysis

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

## Run Week 10 feature engineering

```sh
PYTHONPATH="/Users/mannz/Desktop/polymer degredation/firstdataset/src" \
MPLBACKEND=Agg MPLCONFIGDIR="/tmp/matplotlib-week10" \
python "/Users/mannz/Desktop/polymer degredation/firstdataset/scripts/run_week10_feature_engineering.py"
```

This writes:

- `reports/week10_feature_sets.json`
- `reports/week10_feature_diagnostics.csv`
- `reports/week10_feature_results.csv`
- `reports/week10_feature_comparison.png`
- `WEEK10_FEATURE_ENGINEERING_SUMMARY.txt`

## Run Week 11 feature analysis

```sh
PYTHONPATH="/Users/mannz/Desktop/polymer degredation/firstdataset/src" \
MPLBACKEND=Agg MPLCONFIGDIR="/tmp/matplotlib-week11" \
python "/Users/mannz/Desktop/polymer degredation/firstdataset/scripts/run_week11_feature_analysis.py"
```

This writes:

- `reports/week11_feature_rankings.csv`
- `reports/week11_feature_set_results.csv`
- `reports/week11_feature_set_diagnostics.csv`
- `reports/week11_feature_set_generalization.csv`
- `reports/week11_feature_sets.json`
- `reports/week11_charts/`
- `WEEK11_FEATURE_ANALYSIS_SUMMARY.txt`

## Run Week 12 uncertainty analysis

```sh
PYTHONPATH="/Users/mannz/Desktop/polymer degredation/firstdataset/src" \
MPLBACKEND=Agg MPLCONFIGDIR="/tmp/matplotlib-week12" \
python "/Users/mannz/Desktop/polymer degredation/firstdataset/scripts/run_week12_uncertainty_analysis.py"
```

This writes:

- `reports/week12_prediction_level_uncertainty.csv`
- `reports/week12_uncertainty_metrics.csv`
- `reports/week12_selective_prediction.csv`
- `reports/week12_cross_env_uncertainty.csv`
- `reports/week12_uncertainty_summary.txt`
- `reports/week12_charts/`
