from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from firstdataset.data import CURATED_DATA_PATH, DEFAULT_DATA_PATH, build_curated_qsar_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
METADATA_PATH = REPORTS_DIR / "week4_dataset_metadata.json"
REPORT_PATH = REPORTS_DIR / "week4_dataset_curation.md"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    curated = build_curated_qsar_dataset()

    metadata = {
        "dataset_name": "qsar_biodegradation",
        "dataset_version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(DEFAULT_DATA_PATH),
        "curated_path": str(CURATED_DATA_PATH),
        "row_count": int(curated.shape[0]),
        "feature_count": 41,
        "target_column": "biodegradation_class_id",
        "target_labels": {
            "1": "not_readily_biodegradable",
            "2": "readily_biodegradable",
        },
        "missing_values": int(curated.isna().sum().sum()),
        "representation_status": {
            "bigsmiles_available": False,
            "polymer_identity_available": False,
            "descriptor_columns_available": True,
        },
        "metadata_decisions": [
            "Mapped source columns V1..V41 to the published molecular descriptor names from the QSAR biodegradation dataset documentation.",
            "Added a stable sample_id column for row-level traceability in analysis and reporting.",
            "Kept original class IDs and added human-readable label columns.",
            "Retained all rows because the downloaded file contains no missing values.",
            "Recorded that polymer structures and BigSMILES strings are absent from the source dataset.",
        ],
        "limitations": [
            "This is not a polymer-specific dataset.",
            "No molecular strings, polymer names, or BigSMILES representations are present.",
            "The target is binary class membership, not a continuous degradation rate constant.",
        ],
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")

    report = f"""# Week 4 Dataset Curation and Representation

## Scope
This project currently uses the Kaggle `muhammetvarl/qsarbiodegradation` dataset as the first working dataset version.

## Dataset Version
- Version: `v1`
- Source file: `{DEFAULT_DATA_PATH}`
- Curated file: `{CURATED_DATA_PATH}`
- Rows: `{curated.shape[0]}`
- Descriptor columns: `41`
- Missing values after curation: `{int(curated.isna().sum().sum())}`

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
"""
    REPORT_PATH.write_text(report)

    print(f"Wrote curated dataset to {CURATED_DATA_PATH}")
    print(f"Wrote metadata to {METADATA_PATH}")
    print(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
