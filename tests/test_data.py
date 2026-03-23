from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from firstdataset.data import (
    CURATED_DATA_PATH,
    DEFAULT_DATA_PATH,
    TARGET_COLUMN,
    build_curated_qsar_dataset,
    load_qsar_biodegradation,
    load_tabular_regression_dataset,
    split_qsar_biodegradation,
    split_tabular_regression_dataset,
)
from firstdataset.modeling import run_regression_baselines_from_csv
from firstdataset.modeling import run_qsar_fnn_classifier
from firstdataset.week7_gnn import run_week7_descriptor_graph_prototype


class QSARDataTests(unittest.TestCase):
    def test_default_dataset_path_exists(self) -> None:
        self.assertTrue(Path(DEFAULT_DATA_PATH).exists())

    def test_load_qsar_biodegradation_shapes(self) -> None:
        bundle = load_qsar_biodegradation()
        self.assertEqual(bundle.frame.shape, (1055, 42))
        self.assertEqual(bundle.X.shape, (1055, 41))
        self.assertEqual(bundle.y.shape, (1055,))
        self.assertNotIn(TARGET_COLUMN, bundle.X.columns)
        self.assertEqual(set(bundle.y.astype(str).unique()), {"NRB", "RB"})

    def test_split_qsar_biodegradation_is_stratified(self) -> None:
        split = split_qsar_biodegradation(test_size=0.2, random_state=7)
        self.assertEqual(split.X_train.shape, (844, 41))
        self.assertEqual(split.X_test.shape, (211, 41))
        self.assertEqual(set(split.y_train.astype(str).unique()), {"NRB", "RB"})
        self.assertEqual(set(split.y_test.astype(str).unique()), {"NRB", "RB"})

    def test_build_curated_qsar_dataset(self) -> None:
        curated = build_curated_qsar_dataset(output_path=CURATED_DATA_PATH)
        self.assertEqual(curated.shape, (1055, 45))
        self.assertEqual(curated.columns[0], "sample_id")
        self.assertEqual(curated.iloc[0]["sample_id"], "qsar_00001")
        self.assertIn("SpMax_L", curated.columns)
        self.assertIn("nX", curated.columns)
        self.assertIn("biodegradation_class_id", curated.columns)
        self.assertIn("biodegradation_outcome", curated.columns)

    def test_generic_regression_pipeline(self) -> None:
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "synthetic_regression.csv"
            frame = pd.DataFrame(
                {
                    "f1": rng.normal(size=120),
                    "f2": rng.normal(size=120),
                    "f3": rng.normal(size=120),
                }
            )
            frame["target"] = 3.0 * frame["f1"] - 1.5 * frame["f2"] + 0.5 * frame["f3"]
            frame.to_csv(csv_path, index=False)

            bundle = load_tabular_regression_dataset(csv_path, target_column="target")
            self.assertEqual(bundle.X.shape, (120, 3))
            self.assertEqual(bundle.target_column, "target")

            split = split_tabular_regression_dataset(csv_path, target_column="target")
            self.assertEqual(split.X_train.shape[1], 3)

            results = run_regression_baselines_from_csv(str(csv_path), target_column="target")
            metric_names = {metric for result in results for metric in result.metrics}
            self.assertEqual(metric_names, {"mae", "rmse", "r2"})
            self.assertEqual(len(results), 2)

    def test_qsar_fnn_classifier(self) -> None:
        result = run_qsar_fnn_classifier(random_state=42)
        self.assertEqual(result.model_name, "feedforward_neural_network")
        self.assertEqual(set(result.metrics), {"accuracy", "precision", "recall", "f1_score", "roc_auc"})
        self.assertEqual(len(result.confusion_matrix), 2)
        self.assertEqual(len(result.confusion_matrix[0]), 2)

    def test_week7_descriptor_graph_prototype(self) -> None:
        result = run_week7_descriptor_graph_prototype(random_state=42)
        self.assertEqual(result.model_name, "descriptor_graph_neural_network_prototype")
        self.assertEqual(set(result.metrics), {"accuracy", "precision", "recall", "f1_score", "roc_auc"})
        self.assertEqual(result.graph_info["num_nodes"], 41)
        self.assertEqual(len(result.confusion_matrix), 2)


if __name__ == "__main__":
    unittest.main()
