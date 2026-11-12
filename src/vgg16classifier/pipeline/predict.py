from pathlib import Path

import numpy as np
from vgg16classifier.components.prepare_base_model import (
    CircleMask,
    VGG16Preprocessing,
)
from vgg16classifier.entity.configaration_entities import TrainingConfig
from vgg16classifier.efficientnetv2s.model import GRAD_CAM_LAYER

def binary_class_index(raw_predictions: object) -> int:
    predictions = np.asarray(raw_predictions)
    if predictions.shape == (1, 1):
        return int(predictions[0, 0] >= 0.5)
    if predictions.shape == (1, 2):
        return int(np.argmax(predictions[0]))
    raise ValueError(
        f"Expected one or two binary class outputs, got shape {predictions.shape}"
    )


class PredictionPipeline:
    def __init__(self, filename: str, model_path: Path | None = None):
        self.filename = filename
        self.model_path = model_path

    def _get_model_path(
        self,
        training_config: TrainingConfig | None = None,
    ) -> Path:
        """Use the configured training artifact unless a path was injected."""
        model_path = self.model_path
        if model_path is None:
            if training_config is None:
                from vgg16classifier.config.configuration import (
                    ConfigurationManager,
                )

                training_config = ConfigurationManager().get_training_config()
            model_path = training_config.trained_model_path
        if model_path.suffix.lower() != ".keras":
            raise ValueError(
                f"Expected a native Keras .keras model, got: {model_path}"
            )
        return model_path

    def predict(self):
        from vgg16classifier.config.configuration import ConfigurationManager

        configuration = ConfigurationManager()
        training_config = configuration.get_training_config()
        model_path = self._get_model_path(training_config)
        self.model_path = model_path

        # Load TensorFlow only when inference is requested, not for metric helpers.
        import tensorflow as tf
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "preprocess_input": (
                    tf.keras.applications.vgg16.preprocess_input
                ),
                "CircleMask": CircleMask,
                "VGG16Preprocessing": VGG16Preprocessing,
            },
        )

        imagename = self.filename
        test_image = tf.keras.utils.load_img(
            imagename,
            target_size=tuple(training_config.params_image_size[:2]),
        )
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image, verbose=0)
        class_index = binary_class_index(result)

        if class_index == 1:
            prediction = 'non_cancerous'
        else:
            prediction = 'cancerous'

        from vgg16classifier.components.grad_cam import (
            encode_grad_cam_overlay,
            grad_cam,
        )

        model_architecture = training_config.params_model_architecture
        grad_cam_layer = (
            GRAD_CAM_LAYER
            if model_architecture == "efficientnetv2s"
            else "block5_conv3"
        )
        heatmap = grad_cam(
            model,
            test_image[0],
            class_index=class_index,
            layer_name=grad_cam_layer,
        )
        peak_cell = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        overlay = encode_grad_cam_overlay(test_image[0], heatmap)
        return [
            {
                "image": prediction,
                "grad_cam_peak_cell": [int(row) for row in peak_cell],
                "model_path": str(self.model_path),
                "imagename": imagename,
            },
            {"image": overlay},
        ]