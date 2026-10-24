from pathlib import Path

import numpy as np
import tensorflow as tf

from vgg16classifier.entity.configaration_entities import EvaluationConfigs
from vgg16classifier.utils.common import save_json


def calculate_binary_metrics(
    true_indices: np.ndarray,
    predicted_indices: np.ndarray,
    positive_index: int,
) -> dict[str, int | float]:
    true_labels = np.asarray(true_indices, dtype=np.int64).reshape(-1)
    predicted_labels = np.asarray(predicted_indices, dtype=np.int64).reshape(-1)
    if true_labels.size == 0 or true_labels.shape != predicted_labels.shape:
        raise ValueError("True and predicted labels must be non-empty and equal length")
    if positive_index not in (0, 1):
        raise ValueError("positive_index must be 0 or 1")
    if not np.isin(true_labels, (0, 1)).all() or not np.isin(
        predicted_labels, (0, 1)
    ).all():
        raise ValueError("Binary labels must contain only class indices 0 and 1")

    actual_positive = true_labels == positive_index
    predicted_positive = predicted_labels == positive_index
    true_positive = int(np.sum(actual_positive & predicted_positive))
    false_negative = int(np.sum(actual_positive & ~predicted_positive))
    true_negative = int(np.sum(~actual_positive & ~predicted_positive))
    false_positive = int(np.sum(~actual_positive & predicted_positive))
    positive_support = true_positive + false_negative
    negative_support = true_negative + false_positive
    if positive_support == 0 or negative_support == 0:
        raise ValueError("Evaluation set must contain both binary classes")

    return {
        "sensitivity": true_positive / positive_support,
        "specificity": true_negative / negative_support,
        "true_positive": true_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_positive": false_positive,
    }


class Evaluation:
    POSITIVE_CLASS = "affected"
    CLASS_ORDER = ["affected", "Healthy"]
    THRESHOLD = 0.5

    def __init__(self, config: EvaluationConfigs):
        self.config = config

    def _valid_generator(self, class_mode: str):
        datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            validation_split=0.30
        )
        self.valid_generator = datagenerator.flow_from_directory(
            directory=self.config.training_data,
            classes=self.CLASS_ORDER,
            subset="validation",
            shuffle=False,
            class_mode=class_mode,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        model = self.load_model(self.config.path_of_model)
        if isinstance(model.output_shape, list) or len(model.output_shape) != 2:
            raise ValueError("Expected a single-output binary classification model")
        output_width = model.output_shape[-1]
        if output_width == 1:
            class_mode = "binary"
        elif output_width == 2:
            class_mode = "categorical"
        else:
            raise ValueError(
                f"Expected one or two binary outputs, got {model.output_shape}"
            )

        self._valid_generator(class_mode=class_mode)
        expected_class_indices = {"affected": 0, "Healthy": 1}
        if self.valid_generator.class_indices != expected_class_indices:
            raise ValueError(
                "Evaluation class folders do not match the configured binary mapping: "
                f"{self.valid_generator.class_indices}"
            )

        evaluation_results = model.evaluate(
            self.valid_generator,
            verbose=0,
            return_dict=True,
        )
        raw_predictions = np.asarray(
            model.predict(self.valid_generator, verbose=0)
        )
        if output_width == 1:
            predicted_indices = (
                raw_predictions.reshape(-1) >= self.THRESHOLD
            ).astype(np.int64)
        else:
            predicted_indices = np.argmax(raw_predictions, axis=1)

        accuracy = evaluation_results.get(
            "accuracy", evaluation_results.get("binary_accuracy")
        )
        if accuracy is None:
            raise ValueError("The compiled model did not report an accuracy metric")

        binary_metrics = calculate_binary_metrics(
            true_indices=self.valid_generator.classes,
            predicted_indices=predicted_indices,
            positive_index=self.valid_generator.class_indices[self.POSITIVE_CLASS],
        )
        self.score = {
            "loss": float(evaluation_results["loss"]),
            "accuracy": float(accuracy),
            **binary_metrics,
            "positive_class": self.POSITIVE_CLASS,
            "threshold": self.THRESHOLD,
            "class_indices": self.valid_generator.class_indices,
        }

    def save_score(self):
        save_json(path=Path("evaluation_scores_com726.json"), data=self.score)
