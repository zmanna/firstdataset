# Architecture

## Purpose

This repository is a reproducible machine learning workflow for polymer degradation pathway prediction using curated molecular descriptor data. The project is organized so a reader can trace the work from raw data through final model selection without relying on notebook state or local machine paths.

## System Boundaries

| Area | Responsibility |
|---|---|
| `data/` | Stores source and curated datasets. |
| `src/firstdataset/` | Contains reusable project logic. |
| `scripts/` | Provides command-line experiment runners. |
| `reports/weekXX/` | Stores generated metrics, CSV outputs, and charts. |
| `docs/weekly/` | Stores human-readable experiment summaries. |
| `tests/` | Verifies core data/model workflow behavior. |

## Data Flow

```text
source descriptor CSV
  -> data curation
  -> curated descriptor table
  -> train/test and validation splits
  -> baseline models
  -> neural-network and descriptor-graph prototypes
  -> cross-environment validation
  -> feature engineering and selection
  -> uncertainty analysis
  -> final reliability scoreboard
```

## Module Map

| Module | Role |
|---|---|
| `data.py` | Loads source data, standardizes descriptor columns, creates curated datasets, and produces reproducible splits. |
| `modeling.py` | Runs baseline classification, FNN classification, and generic regression baselines. |
| `week7_gnn.py` | Implements a descriptor-graph prototype adapted to descriptor-only data. |
| `week8_validation.py` | Builds proxy environments and runs cross-environment validation. |
| `week9_validation.py` | Runs stratified cross-validation and optional SMOTE comparison. |
| `week10_features.py` | Builds baseline and chemistry-aware proxy feature bundles. |
| `week11_analysis.py` | Ranks features and evaluates selected feature sets. |
| `week12_uncertainty_analysis.py` | Computes prediction uncertainty, calibration, selective prediction, and cross-environment uncertainty behavior. |
| `week13_model_selection.py` | Produces the final candidate scoreboard and model-selection report. |

## Output Convention

New generated artifacts should go into `reports/weekXX/`. Human-readable summaries should go into `docs/weekly/`. Do not add generated reports to the repository root.

## Current Constraints

- The current source data is descriptor-only.
- The source dataset name still references QSAR biodegradation; the project goal is polymer degradation pathway prediction.
- The current target is class-based, not a continuous degradation-rate constant.
- Polymer-specific identities, repeat units, and pathway labels should be added in future dataset extensions.

