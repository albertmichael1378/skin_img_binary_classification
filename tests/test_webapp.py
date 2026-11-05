import base64
from io import BytesIO
import unittest
from unittest.mock import Mock, patch

from PIL import Image

import webapp


class PredictionRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = webapp.app.test_client()

    @staticmethod
    def _encoded_test_image():
        image_bytes = BytesIO()
        Image.new("RGB", (4, 4), color="red").save(image_bytes, format="JPEG")
        return base64.b64encode(image_bytes.getvalue()).decode("ascii")

    def test_predict_rejects_missing_json_payload(self):
        response = self.client.post("/predict", json=None)

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_predict_rejects_invalid_base64(self):
        response = self.client.post("/predict", json={"image": "not base64!"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("valid base64", response.get_json()["error"])

    def test_predict_rejects_non_image_payload(self):
        response = self.client.post(
            "/predict",
            json={"image": base64.b64encode(b"not an image").decode("ascii")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("valid supported image", response.get_json()["error"])

    @patch("webapp.PredictionPipeline")
    def test_predict_uses_isolated_image_and_returns_result(self, pipeline_class):
        expected_result = [{"image": "cancerous"}, {"image": "overlay"}]
        image_bytes = base64.b64decode(self._encoded_test_image())
        pipeline = Mock()
        pipeline.predict.return_value = expected_result
        pipeline_class.return_value = pipeline

        def verify_temporary_image(image_path):
            with open(image_path, "rb") as image_file:
                self.assertEqual(image_file.read(), image_bytes)
            return pipeline

        pipeline_class.side_effect = verify_temporary_image
        response = self.client.post(
            "/predict",
            json={"image": base64.b64encode(image_bytes).decode("ascii")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), expected_result)
        pipeline_class.assert_called_once()

    @patch("webapp.PredictionPipeline")
    def test_predict_logs_and_returns_generic_error_when_inference_fails(
        self, pipeline_class
    ):
        pipeline_class.return_value.predict.side_effect = RuntimeError("model error")

        with self.assertLogs("vgg16ClassifierLogger", level="ERROR"):
            response = self.client.post(
                "/predict",
                json={"image": self._encoded_test_image()},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"error": "Prediction failed"})


if __name__ == "__main__":
    unittest.main()
