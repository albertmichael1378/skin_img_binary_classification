from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vgg16classifier.entity.configaration_entities import EvaluationConfigs

if TYPE_CHECKING:
    # Keep TensorFlow as a type-only dependency so metric tests can run without it.
    import tensorflow as tf


def _validated_labels(
    true_indices: np.ndarray,
    predicted_indices: np.ndarray,
    positive_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    true_values = np.asarray(true_indices).reshape(-1)
    predicted_values = np.asarray(predicted_indices).reshape(-1)
    if true_values.size == 0 or true_values.shape != predicted_values.shape:
        raise ValueError("True and predicted labels must be non-empty and equal length")
    if positive_index not in (0, 1):
        raise ValueError("positive_index must be 0 or 1")
    if not np.isin(true_values, (0, 1)).all() or not np.isin(
        predicted_values, (0, 1)
    ).all():
        raise ValueError("Binary labels must contain only class indices 0 and 1")
    true_labels = true_values.astype(np.int64)
    predicted_labels = predicted_values.astype(np.int64)
    if set(np.unique(true_labels)) != {0, 1}:
        raise ValueError("Evaluation set must contain both binary classes")
    return true_labels, predicted_labels


def _divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _roc_auc(true_positive_labels: np.ndarray, scores: np.ndarray) -> float:
    """Calculate ROC-AUC using average ranks so tied scores are handled correctly."""
    positive_count = int(np.sum(true_positive_labels))
    negative_count = int(true_positive_labels.size - positive_count)
    if not positive_count or not negative_count:
        raise ValueError("ROC-AUC requires positive and negative examples")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end

    positive_rank_sum = float(np.sum(ranks[true_positive_labels]))
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def _average_precision(
    true_positive_labels: np.ndarray, scores: np.ndarray
) -> float:
    """Calculate threshold-independent area under the precision-recall curve."""
    positive_count = int(np.sum(true_positive_labels))
    if not positive_count:
        raise ValueError("PR-AUC requires positive examples")

    order = np.argsort(-scores, kind="mergesort")
    ranked_scores = scores[order]
    ranked_labels = true_positive_labels[order].astype(np.int64)
    cumulative_true_positives = np.cumsum(ranked_labels)
    group_ends = np.flatnonzero(np.diff(ranked_scores)) + 1
    group_ends = np.append(group_ends, ranked_labels.size)

    # Advance recall once per distinct score so tied predictions have no
    # arbitrary within-tie ordering.
    previous_true_positives = 0
    average_precision = 0.0
    for group_end in group_ends:
        true_positives = int(cumulative_true_positives[group_end - 1])
        precision = true_positives / group_end
        recall_increment = (
            true_positives - previous_true_positives
        ) / positive_count
        average_precision += precision * recall_increment
        previous_true_positives = true_positives
    return float(average_precision)


def calculate_binary_metrics(
    true_indices: np.ndarray,
    predicted_indices: np.ndarray,
    positive_index: int,
    positive_probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    """Calculate thresholded classification metrics and optional probability metrics."""
    true_labels, predicted_labels = _validated_labels(
        true_indices, predicted_indices, positive_index
    )
    actual_positive = true_labels == positive_index
    predicted_positive = predicted_labels == positive_index
    true_positive = int(np.sum(actual_positive & predicted_positive))
    false_negative = int(np.sum(actual_positive & ~predicted_positive))
    true_negative = int(np.sum(~actual_positive & ~predicted_positive))
    false_positive = int(np.sum(~actual_positive & predicted_positive))

    precision = _divide(true_positive, true_positive + false_positive)
    sensitivity = _divide(true_positive, true_positive + false_negative)
    specificity = _divide(true_negative, true_negative + false_positive)
    f1_score = _divide(2 * precision * sensitivity, precision + sensitivity)
    accuracy = _divide(
        true_positive + true_negative, true_labels.size
    )
    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": sensitivity,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1_score,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "true_positive": true_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "support": int(true_labels.size),
    }

    if positive_probabilities is not None:
        probabilities = np.asarray(
            positive_probabilities, dtype=np.float64
        ).reshape(-1)
        if probabilities.shape != true_labels.shape:
            raise ValueError("Probability and label arrays must have equal length")
        if not np.isfinite(probabilities).all() or np.any(
            (probabilities < 0) | (probabilities > 1)
        ):
            raise ValueError("Class probabilities must be finite and between 0 and 1")
        metrics.update(
            {
                "roc_auc": _roc_auc(actual_positive, probabilities),
                "pr_auc": _average_precision(actual_positive, probabilities),
                "brier_score": float(
                    np.mean((probabilities - actual_positive.astype(float)) ** 2)
                ),
            }
        )

    return metrics


def _class_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: tuple[str, str],
) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    for class_index, class_name in enumerate(class_names):
        true_class = true_labels == class_index
        predicted_class = predicted_labels == class_index
        true_positive = int(np.sum(true_class & predicted_class))
        false_positive = int(np.sum(~true_class & predicted_class))
        false_negative = int(np.sum(true_class & ~predicted_class))
        precision = _divide(true_positive, true_positive + false_positive)
        recall = _divide(true_positive, true_positive + false_negative)
        report[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": _divide(2 * precision * recall, precision + recall),
            "support": int(np.sum(true_class)),
        }
    return report


def _stratified_bootstrap_intervals(
    true_labels: np.ndarray,
    positive_probabilities: np.ndarray,
    *,
    positive_index: int,
    threshold: float,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Estimate uncertainty by resampling images within each class.

    This keeps each class's observed support fixed, but assumes individual
    images are independent; it is not a substitute for patient-level bootstrap.
    """
    # A fixed seed makes the reported intervals reproducible across runs.
    rng = np.random.default_rng(seed)
    class_indices = [
        np.flatnonzero(true_labels == class_index) for class_index in (0, 1)
    ]
    metric_samples: dict[str, list[float]] = {
        key: []
        for key in (
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1_score",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc",
            "brier_score",
        )
    }
    for _ in range(samples):
        sampled_indices = np.concatenate(
            [
                rng.choice(indices, size=indices.size, replace=True)
                for indices in class_indices
            ]
        )
        sampled_labels = true_labels[sampled_indices]
        sampled_probabilities = positive_probabilities[sampled_indices]
        sampled_predictions = np.where(
            sampled_probabilities >= threshold, positive_index, 1 - positive_index
        )
        metrics = calculate_binary_metrics(
            sampled_labels,
            sampled_predictions,
            positive_index,
            sampled_probabilities,
        )
        for key in metric_samples:
            metric_samples[key].append(float(metrics[key]))

    tail_probability = (1 - confidence_level) / 2
    return {
        key: {
            "lower": float(np.quantile(values, tail_probability)),
            "upper": float(np.quantile(values, 1 - tail_probability)),
            "confidence_level": confidence_level,
        }
        for key, values in metric_samples.items()
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_fingerprint(dataset_root: Path, filepaths: list[str]) -> str:
    digest = hashlib.sha256()
    for filepath in sorted(filepaths):
        image_path = Path(filepath)
        relative_path = image_path.resolve().relative_to(dataset_root.resolve())
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(image_path.stat().st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


class Evaluation:
    """Evaluate a saved binary classifier and persist metrics plus image-level results."""

    def __init__(self, config: EvaluationConfigs):
        self.config = config
        self.score: dict[str, Any] | None = None
        self.predictions: list[dict[str, Any]] = []

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        import tensorflow as tf
        from vgg16classifier.components.prepare_base_model import VGG16Preprocessing

        return tf.keras.models.load_model(
            path,
            custom_objects={
                "preprocess_input": tf.keras.applications.vgg16.preprocess_input,
                "VGG16Preprocessing": VGG16Preprocessing,
            },
        )

    def _load_test_dataset(self):
        import tensorflow as tf

        if not self.config.test_data_dir.is_dir():
            raise FileNotFoundError(
                f"Evaluation dataset directory does not exist: "
                f"{self.config.test_data_dir}"
            )
        return tf.keras.utils.image_dataset_from_directory(
            directory=self.config.test_data_dir,
            # Explicit names lock folder labels to the same indices as the model.
            class_names=list(self.config.class_names),
            labels="inferred",
            label_mode="int",
            image_size=tuple(self.config.params_image_size[:-1]),
            batch_size=self.config.params_batch_size,
            # Stable ordering keeps predictions aligned with file_paths in the CSV.
            shuffle=False,
            interpolation="bilinear",
        )

    def evaluation(self) -> None:
        if not np.isfinite(self.config.decision_threshold) or not (
            0 <= self.config.decision_threshold <= 1
        ):
            raise ValueError("decision_threshold must be between 0 and 1")
        if self.config.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be at least 1")
        if not np.isfinite(self.config.confidence_level) or not (
            0 < self.config.confidence_level < 1
        ):
            raise ValueError("confidence_level must be between 0 and 1")
        if len(self.config.class_names) != 2 or len(set(self.config.class_names)) != 2:
            raise ValueError("Exactly two distinct class_names must be configured")
        if self.config.positive_class not in self.config.class_names:
            raise ValueError("positive_class must be one of the configured class names")
        if self.config.params_batch_size < 1:
            raise ValueError("params_batch_size must be at least 1")

        model_path = self.config.path_of_model
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        model = self.load_model(model_path)
        dataset = self._load_test_dataset()
        class_indices = {
            class_name: index for index, class_name in enumerate(self.config.class_names)
        }
        if dataset.class_names != list(self.config.class_names):
            raise ValueError(
                f"Unexpected evaluation class mapping: {dataset.class_names}"
            )
        positive_index = class_indices[self.config.positive_class]

        labels: list[np.ndarray] = []
        class_one_probabilities: list[np.ndarray] = []
        for images, batch_labels in dataset:
            outputs = model(images, training=False)
            if isinstance(outputs, (list, tuple, dict)):
                raise ValueError(
                    "Expected a single-output binary classification model"
                )
            predictions = np.asarray(outputs)
            if predictions.ndim != 2 or predictions.shape[1] not in (1, 2):
                raise ValueError(
                    "Expected model predictions with shape (batch, 1) or (batch, 2), "
                    f"got {predictions.shape}"
                )
            if not np.isfinite(predictions).all() or np.any(
                (predictions < 0) | (predictions > 1)
            ):
                raise ValueError("Model outputs must be finite class probabilities")
            # A sigmoid head models class index 1, as assigned by the loader.
            if predictions.shape[1] == 1:
                batch_class_one_probabilities = predictions[:, 0]
            else:
                if not np.allclose(np.sum(predictions, axis=1), 1, atol=1e-3):
                    raise ValueError(
                        "Two-class model outputs must sum to one per sample"
                    )
                batch_class_one_probabilities = predictions[:, 1]
            labels.append(np.asarray(batch_labels).reshape(-1).astype(np.int64))
            class_one_probabilities.append(
                np.asarray(batch_class_one_probabilities, dtype=np.float64)
            )

        true_labels = np.concatenate(labels)
        class_one_probability = np.concatenate(class_one_probabilities)
        # Use the complement when class index 0 is the configured positive class.
        positive_probabilities = (
            class_one_probability
            if positive_index == 1
            else 1 - class_one_probability
        )
        predicted_labels = np.where(
            positive_probabilities >= self.config.decision_threshold,
            positive_index,
            1 - positive_index,
        )
        metrics = calculate_binary_metrics(
            true_labels,
            predicted_labels,
            positive_index,
            positive_probabilities,
        )
        class_metrics = _class_metrics(
            true_labels, predicted_labels, self.config.class_names
        )
        confusion_matrix = [
            [
                int(np.sum((true_labels == actual) & (predicted_labels == predicted)))
                for predicted in (0, 1)
            ]
            for actual in (0, 1)
        ]
        filepaths = list(dataset.file_paths)
        if len(filepaths) != true_labels.size:
            raise ValueError(
                "Evaluation image paths and labels have different sample counts"
            )

        self.score = {
            "schema_version": 1,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": {
                "path": str(model_path),
                "sha256": _file_sha256(model_path),
            },
            "dataset": {
                "root": str(self.config.test_data_dir),
                "split": "configured_test_data",
                "sample_count": int(true_labels.size),
                "class_indices": class_indices,
                "class_counts": {
                    class_name: int(np.sum(true_labels == index))
                    for class_name, index in class_indices.items()
                },
                "manifest_algorithm": "sha256(relative_path_and_file_size)",
                "manifest_sha256": _dataset_fingerprint(
                    self.config.test_data_dir, filepaths
                ),
                "class_mapping_verified": self.config.class_mapping_verified,
                # This is an explicit user attestation, not an automatic overlap check.
                "dataset_independence_verified": (
                    self.config.dataset_independence_verified
                ),
            },
            "protocol": {
                "positive_class": self.config.positive_class,
                "decision_threshold": self.config.decision_threshold,
                "threshold_selection": "configured_before_evaluation; not tuned here",
                "bootstrap_samples": self.config.bootstrap_samples,
                "confidence_level": self.config.confidence_level,
                "confidence_interval_method": (
                    "stratified image-level percentile bootstrap; "
                    "image independence is not verified"
                ),
            },
            "metrics": metrics,
            "class_metrics": class_metrics,
            "confusion_matrix": {
                "labels": list(self.config.class_names),
                "rows": "true_label",
                "columns": "predicted_label",
                "matrix": confusion_matrix,
            },
            "confidence_intervals": _stratified_bootstrap_intervals(
                true_labels,
                positive_probabilities,
                positive_index=positive_index,
                threshold=self.config.decision_threshold,
                samples=self.config.bootstrap_samples,
                confidence_level=self.config.confidence_level,
                seed=self.config.seed,
            ),
        }
        self.predictions = [
            {
                "image": Path(filepath)
                .resolve()
                .relative_to(self.config.test_data_dir.resolve())
                .as_posix(),
                "true_label": self.config.class_names[int(true_label)],
                "predicted_label": self.config.class_names[int(predicted_label)],
                "positive_class_probability": float(probability),
                "decision_threshold": self.config.decision_threshold,
                "correct": bool(true_label == predicted_label),
            }
            for filepath, true_label, predicted_label, probability in zip(
                filepaths,
                true_labels,
                predicted_labels,
                positive_probabilities,
            )
        ]

    def save_score(self) -> None:
        if self.score is None:
            raise RuntimeError("Run evaluation() before saving evaluation results")

        # Create the configured artifact root and output parents on every entry path.
        self.config.root_dir.mkdir(parents=True, exist_ok=True)
        self.config.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.predictions_output_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        with self.config.metrics_output_path.open(
            "w", encoding="utf-8"
        ) as metrics_file:
            json.dump(self.score, metrics_file, indent=2)
            metrics_file.write("\n")

        with self.config.predictions_output_path.open(
            "w", newline="", encoding="utf-8"
        ) as predictions_file:
            writer = csv.DictWriter(
                predictions_file,
                fieldnames=[
                    "image",
                    "true_label",
                    "predicted_label",
                    "positive_class_probability",
                    "decision_threshold",
                    "correct",
                ],
            )
            writer.writeheader()
            writer.writerows(self.predictions)
