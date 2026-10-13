import json
import tempfile
import unittest
from pathlib import Path

from vgg16classifier.components.run_metadata import (
    create_training_metadata,
    finish_training_metadata,
    write_run_metadata,
)
from vgg16classifier.pipeline.predict import binary_class_index


class TrainingMetadataTests(unittest.TestCase):
    def test_metadata_captures_inventory_labels_and_model_checksum(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "dataset"
            (dataset_root / "cancerous").mkdir(parents=True)
            (dataset_root / "non_cancerous").mkdir(parents=True)
            (dataset_root / "cancerous" / "one.jpg").write_bytes(b"image-a")
            (dataset_root / "non_cancerous" / "two.jpg").write_bytes(b"image-b")

            metadata = create_training_metadata(
                repo_root=root,
                dataset_root=dataset_root,
                train_filenames=["cancerous/one.jpg"],
                train_labels=[0],
                validation_filenames=["non_cancerous/two.jpg"],
                validation_labels=[1],
                class_indices={"cancerous": 0, "non_cancerous": 1},
                parameters={"epochs": 2, "batch_size": 1},
                preprocessing={"rescale": None},
                optimizer="SGD",
                loss="binary_crossentropy",
                model_path=root / "model.h5",
            )

            model_path = root / "model.h5"
            model_path.write_bytes(b"model")
            finish_training_metadata(
                metadata,
                status="completed",
                history={"loss": [0.8, 0.4]},
                model_path=model_path,
            )
            metadata_path = root / "run_metadata.json"
            write_run_metadata(metadata_path, metadata)
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["dataset"]["sample_count"], 2)
            self.assertEqual(
                saved["dataset"]["class_counts"],
                {"cancerous": 1, "non_cancerous": 1},
            )
            self.assertEqual(
                saved["dataset"]["split_class_counts"]["training"],
                {"cancerous": 1, "non_cancerous": 0},
            )
            self.assertEqual(len(saved["dataset"]["manifest_sha256"]), 64)
            self.assertEqual(len(saved["model"]["sha256"]), 64)
            self.assertEqual(saved["training"]["history"]["loss"], [0.8, 0.4])

    def test_binary_output_uses_sigmoid_threshold_or_two_class_argmax(self):
        self.assertEqual(binary_class_index([[0.49]]), 0)
        self.assertEqual(binary_class_index([[0.5]]), 1)
        self.assertEqual(binary_class_index([[0.1, 0.9]]), 1)

    def test_binary_output_rejects_unexpected_shapes(self):
        with self.assertRaises(ValueError):
            binary_class_index([[0.2, 0.3, 0.5]])

    def test_failed_run_metadata_records_the_error(self):
        from vgg16classifier.components.run_metadata import finish_training_metadata

        metadata = {
            "training": {"history": {}, "error": None},
            "model": {"sha256": None},
        }
        finish_training_metadata(
            metadata,
            status="failed",
            error=RuntimeError("training failed"),
        )

        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(
            metadata["training"]["error"],
            {"type": "RuntimeError", "message": "training failed"},
        )


if __name__ == "__main__":
    unittest.main()
