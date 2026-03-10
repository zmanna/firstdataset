from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "data" / "qsarbiodegradation" / "qsar-biodeg.csv"
)
CURATED_DATA_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "qsar_biodegradation_curated.csv"
)
TARGET_COLUMN: Final[str] = "Class"
TARGET_LABELS: Final[dict[int, str]] = {1: "NRB", 2: "RB"}
TARGET_LABEL_NAMES: Final[dict[int, str]] = {
    1: "not_readily_biodegradable",
    2: "readily_biodegradable",
}
FEATURE_COLUMNS: Final[list[str]] = [f"V{i}" for i in range(1, 42)]
STANDARDIZED_FEATURE_COLUMNS: Final[list[str]] = [
    "SpMax_L",
    "J_Dz_e",
    "nHM",
    "F01_N_N",
    "F04_C_N",
    "NssssC",
    "nCb_minus",
    "C_percent",
    "nCp",
    "nO",
    "F03_C_N",
    "SdssC",
    "HyWi_B_m",
    "LOC",
    "SM6_L",
    "F03_C_O",
    "Me",
    "Mi",
    "nN_N",
    "nArNO2",
    "nCRX3",
    "SpPosA_B_p",
    "nCIR",
    "B01_C_Br",
    "B03_C_Cl",
    "N_073",
    "SpMax_A",
    "Psi_i_1d",
    "B04_C_Br",
    "SdO",
    "TI2_L",
    "nCrt",
    "C_026",
    "F02_C_N",
    "nHDon",
    "SpMax_B_m",
    "Psi_i_A",
    "nN",
    "SM6_B_m",
    "nArCOOR",
    "nX",
]


@dataclass(frozen=True)
class QSARDataBundle:
    X: pd.DataFrame
    y: pd.Series
    frame: pd.DataFrame
    target_column: str = TARGET_COLUMN


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_tabular_regression_dataset(
    csv_path: str | Path,
    *,
    target_column: str,
    drop_missing: bool = True,
) -> QSARDataBundle:
    """Load a generic tabular regression dataset with a numeric target column."""
    dataset_path = Path(csv_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    frame = pd.read_csv(dataset_path)
    if target_column not in frame.columns:
        raise ValueError(f"Expected target column '{target_column}' in {dataset_path}")

    frame = frame.apply(pd.to_numeric, errors="raise")
    if drop_missing:
        frame = frame.dropna(axis=0).reset_index(drop=True)

    y = frame[target_column].astype("float64")
    X = frame.drop(columns=[target_column]).astype("float64")
    return QSARDataBundle(X=X, y=y, frame=frame, target_column=target_column)


def split_tabular_regression_dataset(
    csv_path: str | Path,
    *,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_missing: bool = True,
) -> DatasetSplit:
    """Return a reproducible train/test split for a generic regression dataset."""
    bundle = load_tabular_regression_dataset(
        csv_path,
        target_column=target_column,
        drop_missing=drop_missing,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=test_size,
        random_state=random_state,
    )
    return DatasetSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def standardize_qsar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename anonymous descriptor columns into a stable project schema."""
    rename_map = dict(zip(FEATURE_COLUMNS, STANDARDIZED_FEATURE_COLUMNS))
    rename_map[TARGET_COLUMN] = "biodegradation_class_id"
    standardized = frame.rename(columns=rename_map).copy()
    standardized.insert(
        0,
        "sample_id",
        [f"qsar_{idx:05d}" for idx in range(1, len(standardized) + 1)],
    )
    standardized["biodegradation_class_label"] = standardized["biodegradation_class_id"].map(TARGET_LABELS)
    standardized["biodegradation_outcome"] = standardized["biodegradation_class_id"].map(TARGET_LABEL_NAMES)
    return standardized


def load_qsar_biodegradation(
    csv_path: str | Path = DEFAULT_DATA_PATH,
    *,
    drop_missing: bool = True,
    target_as_category: bool = True,
) -> QSARDataBundle:
    """Load the QSAR biodegradation dataset into feature/target objects."""
    dataset_path = Path(csv_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    frame = pd.read_csv(dataset_path)

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {dataset_path}")

    frame = frame.apply(pd.to_numeric, errors="raise")

    if drop_missing:
        frame = frame.dropna(axis=0).reset_index(drop=True)

    y = frame[TARGET_COLUMN].astype("int64")
    if target_as_category:
        y = y.map(TARGET_LABELS).astype("category")

    X = frame.drop(columns=[TARGET_COLUMN]).astype("float64")

    return QSARDataBundle(X=X, y=y, frame=frame, target_column=TARGET_COLUMN)


def split_qsar_biodegradation(
    csv_path: str | Path = DEFAULT_DATA_PATH,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    drop_missing: bool = True,
    target_as_category: bool = True,
) -> DatasetSplit:
    """Return a reproducible train/test split for downstream model evaluation."""
    bundle = load_qsar_biodegradation(
        csv_path,
        drop_missing=drop_missing,
        target_as_category=target_as_category,
    )
    stratify_labels = bundle.y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    return DatasetSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def build_curated_qsar_dataset(
    csv_path: str | Path = DEFAULT_DATA_PATH,
    *,
    output_path: str | Path = CURATED_DATA_PATH,
) -> pd.DataFrame:
    """Create a curated dataset version with stable column names and labels."""
    bundle = load_qsar_biodegradation(csv_path, target_as_category=False)
    curated = standardize_qsar_columns(bundle.frame)
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    curated.to_csv(target_path, index=False)
    return curated
