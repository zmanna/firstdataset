# Week 4 Dataset Curation and Representation

## Scope
This project currently uses the Kaggle `muhammetvarl/qsarbiodegradation` dataset as the first working dataset version.

## Dataset Version
- Version: `v1`
- Source file: `/Users/mannz/Desktop/polymer degredation/firstdataset/data/qsarbiodegradation/qsar-biodeg.csv`
- Curated file: `/Users/mannz/Desktop/polymer degredation/firstdataset/data/processed/qsar_biodegradation_curated.csv`
- Rows: `1055`
- Descriptor columns: `41`
- Missing values after curation: `0`

## Representation Decisions
- Mapped descriptor columns from `V1..V41` to their published molecular descriptor names such as `SpMax_L`, `J_Dz_e`, and `nX`.
- Added a stable row identifier column `sample_id` with values like `qsar_00001`.
- Preserved the source class ID as `biodegradation_class_id`.
- Added `biodegradation_class_label` and `biodegradation_outcome` for readability in downstream analysis.
- Kept a flat tabular descriptor representation because no structure strings are present.

## BigSMILES Status
BigSMILES standardization is blocked for this dataset. The source file contains only anonymous descriptors and a class label, with no polymer names, repeat units, SMILES, or BigSMILES strings to standardize.

## Metadata Decisions
- Accept the Kaggle file as the first dataset version for pipeline development.
- Treat the dataset as descriptor-only biodegradation data.
- Preserve all rows because no missing values are present.
- Record representation limitations explicitly instead of fabricating polymer structure fields.

## Recommendation
Use this dataset for pipeline and classification prototyping. Acquire a polymer-specific dataset with polymer identities and continuous degradation-rate targets before claiming BigSMILES coverage or regression-on-rate deliverables.
