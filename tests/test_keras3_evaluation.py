import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vgg16classifier.components.evaluation_keras3 import (
    Keras3Evaluation,
    _keras_builtin_metrics,
    _selected_validation_roc,
    validate_model_keras_version,
)
from vgg16classifier.entity.configaration_entities import EvaluationConfigs


class _EvaluationDataset:
    class_names = ["negative", "positive"]

    def __init__(self, root: Path):
        self.file_paths = [
            str(root / "negative" / "sample.jpg"),
            str(root / "positive" / "sample.jpg"),
        ]

    def __iter__(self):
        yield np.array([[0.1], [0.9]]), np.array([0, 1])


class _EvaluationModel:
    def __call__(self, images, training=False):
        if training:
            raise AssertionError("Evaluation must run with training=False")
        return images


class Keras3ModelCompatibilityTests(unittest.TestCase):
    def _create_model_archive(self, path: Path, saved_version: str) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "metadata.json",
                json.dumps({"keras_version": saved_version}),
            )

    def test_rejects_runtime_older_than_model_serialization_version(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.keras"
            self._create_model_archive(model_path, "3.13.2")

            with self.assertRaisesRegex(
                RuntimeError,
                "saved with Keras 3.13.2.*active runtime is Keras 3.0.0",
            ):
                validate_model_keras_version(model_path, "3.0.0")

    def test_accepts_runtime_matching_model_serialization_version(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.keras"
            self._create_model_archive(model_path, "3.13.2")

            validate_model_keras_version(model_path, "3.13.2")

    def test_rejects_archive_without_keras_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.keras"
            with zipfile.ZipFile(model_path, "w") as archive:
                archive.writestr("config.json", "{}")

            with self.assertRaisesRegex(ValueError, "no root metadata.json"):
                validate_model_keras_version(model_path, "3.13.2")

    def test_keras_builtin_metrics_use_configured_decision_threshold(self):
        metrics = _keras_builtin_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0.1, 0.6, 0.75, 0.95]),
            threshold=0.7,
        )

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["roc"], 1.0, places=2)
        self.assertGreater(metrics["loss"], 0.0)

    def test_validation_roc_comes_from_metadata_matching_the_model_hash(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_path = root / "model.keras"
            model_path.write_bytes(b"saved model")
            metadata_dir = root / "run_metadata"
            metadata_dir.mkdir()
            metadata = {
                "status": "completed",
                "started_at_utc": "2026-10-27T12:00:00+00:00",
                "training": {
                    "parameters": {"selected_checkpoint_val_auc": 0.91}
                },
                "model": {
                    "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()
                },
            }
            (metadata_dir / "run_meta_data.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            self.assertEqual(
                _selected_validation_roc(metadata_dir, model_path),
                0.91,
            )

    def test_validation_roc_requires_matching_training_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_path = root / "model.keras"
            model_path.write_bytes(b"saved model")
            metadata_dir = root / "run_metadata"
            metadata_dir.mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "No completed training"):
                _selected_validation_roc(metadata_dir, model_path)

    def test_evaluation_report_contains_requested_keras_builtin_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_root = root / "test-data"
            for class_name in ("negative", "positive"):
                image_path = data_root / class_name / "sample.jpg"
                image_path.parent.mkdir(parents=True)
                image_path.write_bytes(b"image")

            model_path = root / "model.keras"
            model_path.write_bytes(b"saved model")
            metadata_dir = root / "run_metadata"
            metadata_dir.mkdir()
            metadata = {
                "status": "completed",
                "started_at_utc": "2026-10-27T12:00:00+00:00",
                "training": {
                    "parameters": {"selected_checkpoint_val_auc": 0.91}
                },
                "model": {
                    "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()
                },
            }
            (metadata_dir / "run_metadata_test.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            config = EvaluationConfigs(
                root_dir=root / "evaluation",
                path_of_model=model_path,
                training_metadata_dir=metadata_dir,
                weights_archive_path=root / "weights.zip",
                test_data_dir=data_root,
                reconstructed_model_path=model_path,
                metrics_output_path=root / "metrics.json",
                predictions_output_path=root / "predictions.csv",
                class_names=("negative", "positive"),
                positive_class="positive",
                decision_threshold=0.7,
                bootstrap_samples=5,
                confidence_level=0.95,
                class_mapping_verified=False,
                dataset_independence_verified=False,
                seed=42,
                params_image_size=[1, 1, 1],
                params_batch_size=2,
            )
            evaluator = Keras3Evaluation(config)

            with (
                patch.object(
                    Keras3Evaluation, "load_model", return_value=_EvaluationModel()
                ),
                patch.object(
                    evaluator,
                    "_load_test_dataset",
                    return_value=_EvaluationDataset(data_root),
                ),
            ):
                evaluator.evaluation()

            self.assertEqual(evaluator.score["metrics"]["accuracy"], 1.0)
            self.assertEqual(evaluator.score["metrics"]["precision"], 1.0)
            self.assertEqual(evaluator.score["metrics"]["recall"], 1.0)
            self.assertAlmostEqual(evaluator.score["metrics"]["roc"], 1.0, places=2)
            self.assertGreater(evaluator.score["metrics"]["loss"], 0.0)
            self.assertEqual(evaluator.score["metrics"]["val_roc"], 0.91)


if __name__ == "__main__":
    unittest.main()
