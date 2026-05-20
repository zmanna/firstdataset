# Data Documentation

## Current Dataset

| Field | Value |
|---|---|
| Source file | `data/qsarbiodegradation/qsar-biodeg.csv` |
| Curated file | `data/processed/qsar_biodegradation_curated.csv` |
| Representation | Tabular molecular descriptors |
| Feature count | 41 descriptor columns |
| Target | Binary class label from source dataset |
| Project use | First modeling substrate for polymer degradation pathway prediction work |

## Curation Decisions

- Source columns `V1..V41` are mapped to stable molecular descriptor names.
- A `sample_id` column is added for row-level traceability.
- Original target IDs are preserved.
- Human-readable target labels are added for downstream analysis.
- Rows are retained when no missing values are present.

## Known Limitations

- The dataset does not include polymer names, repeat units, BigSMILES, or explicit pathway labels.
- The target is not a continuous degradation rate.
- The current data should not be presented as a complete polymer degradation dataset.
- External polymer-specific datasets are needed before making stronger claims about pathway generalization.

## Adding New Data

When adding a new dataset, include:

- source and license
- download or collection date
- target definition
- feature definitions
- preprocessing steps
- known bias or coverage limitations
- whether data can be redistributed publicly

Place raw inputs in a clearly named source folder under `data/`, curated modeling artifacts under `data/processed/`, and data documentation under `docs/`.

