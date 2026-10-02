from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from vgg16classifier.utils.common import decodeImage
from vgg16classifier.pipeline.predict import PredictionPipeline


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)
# Create a global variable for clApp
clApp = ClientApp()


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)

#Below,the code is for the ONNX model prediction, which is blazing fast and can be used in production. The above code is for the Keras model prediction, which is slower and can be used for testing purposes.
def pass_mod():
    
    def predict():
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        
        # 3. Read the image using PIL (Pillow) and resize it to your model's expected size
        # Change 224, 224 to whatever your original model was trained on
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        
        # 4. Convert to a NumPy array and normalize (0 to 1 range)
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # 5. Expand dimensions to create a "batch" of 1: shape becomes (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 6. Run the blazing fast ONNX prediction
        # The output is a list of lists, so we grab [0] to get the array of probabilities
        raw_predictions = session.run(None, {input_name: img_array})[0]
        
        # Example: Get the index of the highest probability disease
        predicted_class_index = np.argmax(raw_predictions)
        
        return jsonify({
            'class_index': int(predicted_class_index),
            'confidence': float(raw_predictions[0][predicted_class_index])
        })

if __name__ == "__main__":
    
    #app.run(host='0.0.0.0', port=8080) #for aws
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

