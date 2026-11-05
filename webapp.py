import base64
import binascii
from io import BytesIO
import os
from tempfile import TemporaryDirectory

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

from vgg16classifier import logger
from vgg16classifier.pipeline.predict import PredictionPipeline


os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object with an 'image' field"}), 400

    encoded_image = payload.get("image")
    if not isinstance(encoded_image, str) or not encoded_image:
        return jsonify(
            {"error": "The 'image' field must contain a base64-encoded image"}
        ), 400

    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, ValueError):
        return jsonify({"error": "The 'image' field is not valid base64"}), 400

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        return jsonify(
            {"error": "The uploaded data is not a valid supported image"}
        ), 400

    # Each request gets a distinct input file, preventing concurrent requests
    # from overwriting one another's image.
    with TemporaryDirectory(prefix="skin-prediction-") as temp_dir:
        image_path = os.path.join(temp_dir, "input.jpg")
        with open(image_path, "wb") as image_file:
            image_file.write(image_bytes)

        try:
            result = PredictionPipeline(image_path).predict()
        except Exception:
            logger.exception("Prediction failed")
            return jsonify({"error": "Prediction failed"}), 500

    return jsonify(result)


if __name__ == "__main__":
    
    #app.run(host='0.0.0.0', port=8080) #for aws
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
