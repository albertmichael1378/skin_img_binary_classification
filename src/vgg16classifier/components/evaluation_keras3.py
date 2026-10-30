"""Keras 3 model loader that reuses the shared evaluation report pipeline."""

from __future__ import annotations

import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from vgg16classifier.components.evaluation import Evaluation, _file_sha256


def _selected_validation_roc(
    metadata_dir: Path,
    model_path: Path,
) -> float:
    """Read the selected checkpoint's validation ROC-AUC from its matching run."""
    if not metadata_dir.is_dir():
        raise FileNotFoundError(
            f"Training metadata directory does not exist: {metadata_dir}"
        )

    model_hash = _file_sha256(model_path)
    
    # 1. First, just find the metadata that matches the model hash
    matching_runs: list[tuple[str, dict]] = []
    for metadata_path in metadata_dir.glob("*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read training metadata: {metadata_path}") from error

        if metadata.get("model", {}).get("sha256") == model_hash:
            matching_runs.append((str(metadata.get("started_at_utc", "")), metadata))

    if not matching_runs:
        # SPECIFIC ERROR: We couldn't even find the metadata for this model file
        raise FileNotFoundError(
            f"No training metadata found for model {model_path} "
            f"(SHA-256 {model_hash}). Was this model trained on this machine?"
        )

    # Get the most recent run
    latest_metadata = max(matching_runs, key=lambda run: run[0])[1]

    # 2. Now check the status specifically
    if latest_metadata.get("status") != "completed":
        # SPECIFIC ERROR: Found the model, but training didn't finish
        raise ValueError(
            f"Found metadata for model {model_path}, but training status is "
            f"'{latest_metadata.get('status')}', not 'completed'."
        )

    # 3. Finally, check the ROC specifically
    validation_roc = latest_metadata.get("training", {}).get("parameters", {}).get("selected_checkpoint_val_auc")
    
    if not (
        isinstance(validation_roc, (int, float)) 
        and math.isfinite(validation_roc) 
        and 0 <= validation_roc <= 1
    ):
        # SPECIFIC ERROR: Found the completed model, but the ROC is missing/invalid
        raise ValueError(
            f"Training metadata for model {model_path} is missing a valid "
            f"'selected_checkpoint_val_auc'. Found value: {validation_roc!r}"
        )

    return float(validation_roc)


def _keras_builtin_metrics(
    true_labels: np.ndarray,
    positive_probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    """Calculate evaluation metrics using standalone Keras losses and metrics."""
    import keras

    labels = np.asarray(true_labels, dtype=np.float32).reshape(-1, 1)
    probabilities = np.asarray(
        positive_probabilities, dtype=np.float32
    ).reshape(-1, 1)
    if labels.size == 0 or labels.shape != probabilities.shape:
        raise ValueError("Keras metric labels and probabilities must have equal length")

    binary_accuracy = keras.metrics.BinaryAccuracy(threshold=threshold)
    precision = keras.metrics.Precision(thresholds=threshold)
    recall = keras.metrics.Recall(thresholds=threshold)
    roc_auc = keras.metrics.AUC(curve="ROC", name="roc")
    for metric in (binary_accuracy, precision, recall, roc_auc):
        metric.update_state(labels, probabilities)

    loss = keras.losses.BinaryCrossentropy()(labels, probabilities)
    return {
        "loss": float(loss.numpy()),
        "accuracy": float(binary_accuracy.result().numpy()),
        "precision": float(precision.result().numpy()),
        "recall": float(recall.result().numpy()),
        "roc": float(roc_auc.result().numpy()),
    }


def _version_tuple(version: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", version)
    if match is None:
        raise ValueError(f"Unrecognized Keras version string: {version!r}")
    return tuple(int(part) for part in match.group(1).split("."))


def validate_model_keras_version(model_path: Path, runtime_version: str) -> None:
    """Reject model archives saved by a newer Keras serialization version."""
    if not zipfile.is_zipfile(model_path):
        raise ValueError(
            f"Expected a native .keras ZIP archive, got: {model_path}"
        )

    with zipfile.ZipFile(model_path) as archive:
        try:
            metadata = json.loads(archive.read("metadata.json"))
        except KeyError as error:
            raise ValueError(
                f"Keras model archive has no root metadata.json: {model_path}"
            ) from error
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Keras model archive metadata is not valid JSON: {model_path}"
            ) from error

    saved_version = metadata.get("keras_version")
    if not isinstance(saved_version, str):
        raise ValueError(
            f"Keras model archive does not declare its Keras version: {model_path}"
        )
    if _version_tuple(runtime_version) < _version_tuple(saved_version):
        raise RuntimeError(
            f"Model was saved with Keras {saved_version}, but the active runtime "
            f"is Keras {runtime_version}. Use Keras {saved_version} or newer to "
            "load this model directly."
        )


class Keras3Evaluation(Evaluation):
    """Evaluate native Keras 3 models with built-in metrics and shared reports."""

    @staticmethod
    def load_model(path: Path) -> Any:
        import keras

        validate_model_keras_version(Path(path), keras.__version__)
        if keras.backend.backend() != "tensorflow":
            raise RuntimeError(
                "The evaluation data pipeline uses TensorFlow; configure Keras "
                "3 to use the TensorFlow backend."
            )
        from vgg16classifier.components.prepare_base_model import VGG16Preprocessing

        return keras.models.load_model(
            path,
            custom_objects={
                "preprocess_input": keras.applications.vgg16.preprocess_input,
                 "VGG16Preprocessing": VGG16Preprocessing,
            },
            compile=False,
            safe_mode=True,
        )

    def _additional_metrics(
        self,
        true_labels: np.ndarray,
        positive_probabilities: np.ndarray,
        positive_index: int,
    ) -> dict[str, float]:
        positive_labels = (true_labels == positive_index).astype(np.int32)
        metrics = _keras_builtin_metrics(
            positive_labels,
            positive_probabilities,
            threshold=self.config.decision_threshold,
        )
        metrics["val_roc"] = _selected_validation_roc(
            self.config.training_metadata_dir,
            self.config.path_of_model,
        )
        return metrics
