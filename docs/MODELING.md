# Modeling Documentation

## Modeling Goal

The project compares model and feature-set candidates for polymer degradation pathway prediction under standard validation, cross-environment validation, and uncertainty-aware reliability scoring.

## Evaluation Stages

| Stage | Script | Output |
|---|---|---|
| Dataset curation | `scripts/curate_qsar_dataset.py` | `reports/week04/`, `docs/weekly/week04-*` |
| Baselines | `scripts/train_week5_baselines.py` | `reports/week05/`, `docs/weekly/week05-*` |
| FNN prototype | `scripts/train_week6_fnn.py` | `reports/week06/`, `docs/weekly/week06-*` |
| Descriptor graph prototype | `scripts/train_week7_gnn.py` | `reports/week07/`, `docs/weekly/week07-*` |
| Cross-environment validation | `scripts/run_week8_cross_environment.py` | `reports/week08/`, `docs/weekly/week08-*` |
| Stratified validation | `scripts/run_week9_validation.py` | `reports/week09/`, `docs/weekly/week09-*` |
| Feature engineering | `scripts/run_week10_feature_engineering.py` | `reports/week10/`, `docs/weekly/week10-*` |
| Feature selection | `scripts/run_week11_feature_analysis.py` | `reports/week11/`, `docs/weekly/week11-*` |
| Uncertainty analysis | `scripts/run_week12_uncertainty_analysis.py` | `reports/week12/`, `docs/weekly/week12-*` |
| Final selection | `scripts/run_week13_model_selection.py` | `reports/week13/`, `docs/weekly/week13-*` |

## Reliability Criteria

Final model selection is not based on accuracy alone. The Week 13 scoreboard considers:

- cross-validation accuracy
- ROC-AUC
- calibration through Brier score, log loss, and expected calibration error
- uncertainty separation between correct and incorrect predictions
- selective prediction performance at reduced coverage
- cross-environment accuracy
- overconfidence under distribution shift

## Reproducibility

Most runners use `random_state=42`. Use the command below before running scripts:

```sh
PYTHONPATH=src python scripts/run_week13_model_selection.py
```

## Interpreting Results

Use `docs/weekly/week13-model-selection-report.md` for the human-readable conclusion and `reports/week13/model_reliability_scoreboard.csv` for the machine-readable candidate ranking.

