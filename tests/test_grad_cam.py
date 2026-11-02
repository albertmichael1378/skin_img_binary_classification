import base64
import io
import unittest

import numpy as np
from PIL import Image

from vgg16classifier.components.grad_cam import encode_grad_cam_overlay


class GradCamOverlayTests(unittest.TestCase):
    def test_overlay_returns_base64_encoded_jpeg_at_source_dimensions(self):
        image = np.full((8, 10, 3), 120, dtype=np.uint8)
        heatmap = np.zeros((2, 2), dtype=np.float32)
        heatmap[0, 0] = 1.0

        encoded_overlay = encode_grad_cam_overlay(image, heatmap)
        overlay_bytes = base64.b64decode(encoded_overlay)

        with Image.open(io.BytesIO(overlay_bytes)) as overlay:
            self.assertEqual(overlay.format, "JPEG")
            self.assertEqual(overlay.size, (10, 8))


if __name__ == "__main__":
    unittest.main()
