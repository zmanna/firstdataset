# Polymer Degradation Pathway Prediction

A Python machine learning pipeline for predicting polymer degradation pathways from curated molecular descriptor data.

## Overview

This is the strongest portfolio-style technical repository in the account. It shows a progression from dataset curation to baseline modeling, neural network experiments, validation, feature engineering, feature importance, uncertainty analysis, and final model selection for polymer degradation pathway prediction.

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
| `data/` | Processed and source dataset folders. |
| `reports/` | Generated metrics, charts, feature rankings, uncertainty reports, and model-selection artifacts. |
| `tests/` | Automated tests for key data functionality. |
| `QSAR_BIODEGRADATION_CURATED.csv` | Curated molecular descriptor dataset artifact used by the modeling workflows. The filename reflects the source dataset label, while the project goal is polymer degradation pathway prediction. |
| `WEEK*_*.md, WEEK*_*.txt, WEEK*_*.json` | Stage-by-stage experiment summaries and metrics. |
| `activate-project.sh` | Local project activation helper. |

## How to Run or Review

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e .
PYTHONPATH=src python scripts/run_week13_model_selection.py
```

## How to Read This Repository

Read this project chronologically: start with dataset curation, then baseline modeling, then validation, feature engineering, uncertainty analysis, and final model selection. The central question is how molecular descriptors can support prediction of polymer degradation pathways.

## Documentation Roadmap

This README is the first cleanup pass. The next documentation pass should add:

- A fuller problem statement or assignment prompt.
- Setup notes for required tools, environment variables, or services.
- Example input and output.
- Screenshots, diagrams, or architecture notes where they clarify the project.
- Testing instructions and known limitations.

## Repository Status

This repository has been renamed and labeled as part of a GitHub organization cleanup. The goal is to make the project understandable to future readers, reviewers, and collaborators.
