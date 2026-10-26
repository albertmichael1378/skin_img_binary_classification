import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vgg16classifier.components.evaluation import (
    Evaluation,
    _stratified_bootstrap_intervals,
    calculate_binary_metrics,
)
from vgg16classifier.entity.configaration_entities import EvaluationConfigs


class _TestDataset:
    class_names = ["affected", "Healthy"]

    def __init__(self, root: Path):
        self.file_paths = [
            str(root / "affected" / "one.jpg"),
            str(root / "Healthy" / "two.jpg"),
        ]

    def __iter__(self):
        yield np.array([[0.1], [0.9]]), np.array([0, 1])


class _TestModel:
    def __call__(self, images, training=False):
        if training:
            raise AssertionError("Evaluation must run with training=False")
        return images


class _TwoClassTestModel:
    def __call__(self, images, training=False):
        if training:
            raise AssertionError("Evaluation must run with training=False")
        class_one_probability = np.asarray(images).reshape(-1)
        return np.column_stack(
            (1 - class_one_probability, class_one_probability)
        )


class BinaryEvaluationMetricsTests(unittest.TestCase):
    def test_sensitivity_and_specificity_use_configured_positive_index(self):
        metrics = calculate_binary_metrics(
            true_indices=[0, 0, 1, 1, 1],
            predicted_indices=[0, 1, 0, 1, 1],
            positive_index=0,
        )

        self.assertEqual(metrics["sensitivity"], 0.5)
        self.assertAlmostEqual(metrics["specificity"], 2 / 3)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["true_negative"], 2)
        self.assertEqual(metrics["false_positive"], 1)

    def test_requires_both_classes_in_evaluation_data(self):
        with self.assertRaisesRegex(ValueError, "both binary classes"):
            calculate_binary_metrics(
                true_indices=[0, 0],
                predicted_indices=[0, 1],
                positive_index=0,
            )

    def test_probability_metrics_cover_ranking_and_calibration(self):
        metrics = calculate_binary_metrics(
            true_indices=[0, 0, 1, 1],
            predicted_indices=[0, 0, 1, 1],
            positive_index=1,
            positive_probabilities=[0.05, 0.25, 0.75, 0.95],
        )

        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["roc_auc"], 1)
        self.assertEqual(metrics["pr_auc"], 1)
        self.assertAlmostEqual(metrics["brier_score"], 0.0325)

    def test_ranking_metrics_handle_tied_probabilities(self):
        metrics = calculate_binary_metrics(
            true_indices=[1, 0],
            predicted_indices=[1, 1],
            positive_index=1,
            positive_probabilities=[0.5, 0.5],
        )

        self.assertEqual(metrics["roc_auc"], 0.5)
        self.assertEqual(metrics["pr_auc"], 0.5)

    def test_bootstrap_intervals_are_reproducible_and_cover_all_core_metrics(self):
        true_labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.6, 0.9])
        intervals = _stratified_bootstrap_intervals(
            true_labels,
            probabilities,
            positive_index=1,
            threshold=0.5,
            samples=50,
            confidence_level=0.95,
            seed=42,
        )

        self.assertEqual(
            set(intervals),
            {
                "accuracy",
                "precision",
                "recall",
                "specificity",
                "f1_score",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc",
                "brier_score",
            },
        )
        self.assertEqual(intervals["accuracy"]["lower"], 1)
        self.assertEqual(intervals["accuracy"]["upper"], 1)

    def test_rejects_invalid_probabilities(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            calculate_binary_metrics(
                true_indices=[0, 1],
                predicted_indices=[0, 1],
                positive_index=1,
                positive_probabilities=[0.2, 1.5],
            )

    def test_rejects_non_integer_binary_labels_without_truncating(self):
        with self.assertRaisesRegex(ValueError, "only class indices"):
            calculate_binary_metrics(
                true_indices=[0.5, 1],
                predicted_indices=[0, 1],
                positive_index=1,
            )

    def test_evaluation_uses_positive_class_probability_and_writes_full_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_root = root / "test-data"
            (data_root / "affected").mkdir(parents=True)
            (data_root / "Healthy").mkdir()
            filepaths = _TestDataset(data_root).file_paths
            for filepath in filepaths:
                Path(filepath).write_bytes(b"image")
            model_path = root / "model.keras"
            model_path.write_bytes(b"model")
            config = EvaluationConfigs(
                root_dir=root / "evaluation",
                path_of_model=model_path,
                weights_archive_path=root / "weights.zip",
                test_data_dir=data_root,
                reconstructed_model_path=model_path,
                metrics_output_path=root / "reports" / "metrics" / "metrics.json",
                predictions_output_path=root / "reports" / "predictions" / "images.csv",
                class_names=("affected", "Healthy"),
                positive_class="affected",
                decision_threshold=0.7,
                bootstrap_samples=20,
                confidence_level=0.95,
                class_mapping_verified=False,
                dataset_independence_verified=False,
                seed=42,
                params_image_size=[2, 2, 3],
                params_batch_size=2,
            )
            evaluator = Evaluation(config)

            for model in (_TestModel(), _TwoClassTestModel()):
                with (
                    patch.object(Evaluation, "load_model", return_value=model),
                    patch.object(
                        evaluator,
                        "_load_test_dataset",
                        return_value=_TestDataset(data_root),
                    ),
                ):
                    evaluator.evaluation()

                self.assertEqual(evaluator.score["metrics"]["accuracy"], 1)
                self.assertEqual(evaluator.score["metrics"]["sensitivity"], 1)
                self.assertEqual(
                    evaluator.score["confusion_matrix"]["matrix"], [[1, 0], [0, 1]]
                )
                self.assertFalse(
                    evaluator.score["dataset"]["class_mapping_verified"]
                )
                self.assertFalse(
                    evaluator.score["dataset"]["dataset_independence_verified"]
                )
                self.assertEqual(
                    [row["predicted_label"] for row in evaluator.predictions],
                    ["affected", "Healthy"],
                )
            evaluator.save_score()
            self.assertTrue(config.root_dir.is_dir())
            self.assertTrue(config.metrics_output_path.is_file())
            self.assertTrue(config.predictions_output_path.is_file())
            self.assertEqual(
                json.loads(
                    config.metrics_output_path.read_text(encoding="utf-8")
                )["metrics"]["accuracy"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
