from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
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
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts", "training", "vgg16model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image, verbose=0)
        class_index = binary_class_index(result)

        if class_index == 1:
            prediction = 'non_cancerous'
            return [{"image": prediction}]
        else:
            prediction = 'cancerous'
            return [{"image": prediction}]