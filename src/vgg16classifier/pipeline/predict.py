from pathlib import Path

import numpy as np


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

    def _get_model_path(self) -> Path:
        """Use the configured training artifact unless a path was injected."""
        model_path = self.model_path
        if model_path is None:
            from vgg16classifier.config.configuration import ConfigurationManager

            model_path = (
                ConfigurationManager().get_training_config().trained_model_path
            )
        if model_path.suffix.lower() != ".keras":
            raise ValueError(
                f"Expected a native Keras .keras model, got: {model_path}"
            )
        return model_path

    def predict(self):
        model_path = self._get_model_path()
        self.model_path = model_path

        # Load TensorFlow only when inference is requested, not for metric helpers.
        import tensorflow as tf
        from vgg16classifier.components.prepare_base_model import VGG16Preprocessing

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "preprocess_input": (
                    tf.keras.applications.vgg16.preprocess_input
                ),
                "VGG16Preprocessing": VGG16Preprocessing,
            },
        )

        imagename = self.filename
        test_image = tf.keras.utils.load_img(imagename, target_size=(224, 224))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image, verbose=0)
        class_index = binary_class_index(result)

        if class_index == 1:
            prediction = 'non_cancerous'
            return [{"image": prediction}]
        else:
            prediction = 'cancerous'
            return [{"image": prediction}]