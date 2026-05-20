# Polymer Degradation Pathway Prediction

A Python machine learning pipeline for predicting polymer degradation pathways from curated molecular descriptor data.

## Overview

This project explores how molecular descriptor data can support polymer degradation pathway prediction. The repository is organized as a reproducible machine learning workflow: data curation, baseline modeling, neural network experiments, validation, feature engineering, feature selection, uncertainty analysis, and final model reliability scoring.

The current dataset is descriptor-based. Some source filenames still reference QSAR biodegradation because they preserve the upstream dataset label, but the project framing and modeling goal are polymer degradation pathway prediction.

## What This Project Demonstrates

- Building a reproducible Python data-science project with source modules, scripts, reports, and tests.
- Curating molecular descriptor data for supervised polymer degradation pathway prediction.
- Comparing baseline models, feature sets, validation strategies, and uncertainty metrics.
- Preserving experiment summaries and generated reports so results can be audited later.
- Using tests to protect core data-loading behavior.

## Project Structure

| Path | Purpose |
|---|---|
| `pyproject.toml` | Python project metadata and dependency configuration. |
| `src/firstdataset/` | Reusable Python package containing data, modeling, feature, validation, and analysis code. |
| `scripts/` | Command-line runners for each experiment stage. |
| `data/raw or data/qsarbiodegradation/` | Source dataset inputs preserved for reproducibility. |
| `data/processed/` | Curated datasets used by modeling workflows. |
| `docs/weekly/` | Human-readable weekly experiment notes, summaries, and model-selection writeups. |
| `reports/weekXX/` | Generated metrics, charts, feature rankings, validation outputs, and reliability artifacts grouped by experiment stage. |
| `tests/` | Automated tests for key data functionality. |
| `activate-project.sh` | Local project activation helper. |

## Formal Documentation

- `docs/ARCHITECTURE.md`: system layout, module responsibilities, and data flow.
- `docs/DATA.md`: dataset provenance, curation decisions, and limitations.
- `docs/MODELING.md`: modeling stages, reliability criteria, and result interpretation.
- `CONTRIBUTING.md`: conventions for extending code, datasets, scripts, and reports.

## Setup

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Common Workflows

Run the final model-selection workflow:

```sh
PYTHONPATH=src python scripts/run_week13_model_selection.py
```

Run the test suite:

```sh
PYTHONPATH=src python -m unittest discover -s tests
```

Regenerate an earlier experiment stage:

```sh
PYTHONPATH=src python scripts/run_week10_feature_engineering.py
```

## How to Read This Repository

Read this project chronologically:

1. `docs/weekly/week04-dataset-curation.md`
2. `docs/weekly/week05-baseline-modeling.md`
3. `docs/weekly/week08-cross-environment-validation.md`
4. `docs/weekly/week10-feature-engineering-summary.txt`
5. `docs/weekly/week11-feature-analysis-summary.txt`
6. `docs/weekly/week12-uncertainty-summary.txt`
7. `docs/weekly/week13-model-selection-report.md`

Then inspect the matching generated artifacts in `reports/weekXX/`.

## Extending the Work

Good next contributions include:

- Adding polymer-specific datasets with polymer identities, repeat units, or degradation pathway labels.
- Expanding the feature set with chemistry-aware, graph-derived, or quantum-inspired descriptors.
- Adding clearer model cards for each selected model.
- Improving test coverage around feature engineering and uncertainty scoring.
- Adding examples that show how to run the pipeline on a new dataset.

## Documentation Roadmap

The next documentation pass should add:

- A formal problem statement.
- Dataset documentation and data provenance notes.
- Architecture diagrams for the pipeline.
- Example input and output.
- Model cards for final candidate models.
- Contributor guidance for adding datasets, scripts, and reports.
