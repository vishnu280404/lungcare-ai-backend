import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# Load the trained DenseNet121 model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'densenet121_lung_opacity_final.keras')

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(image_bytes):
    """
    Preprocesses the image for the DenseNet121 model using keras application's builtin function.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image).astype("float32")
    # Expand dimensions to create batch of size 1
    image_array = np.expand_dims(image_array, axis=0)
    # Apply DenseNet specific preprocessing
    image_array = tf.keras.applications.densenet.preprocess_input(image_array)
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server.'}), 500

    print("=== DEBUG LOG ===")
    print(f"request.files keys: {list(request.files.keys())}")
    print(f"request.form keys: {list(request.form.keys())}")
    print(f"'image' in request.files: {'image' in request.files}")

    if 'image' not in request.files:
        print("Validation Failure: 'image' not in request.files")
        return jsonify({'error': "No 'image' field in the request. Found files keys: " + str(list(request.files.keys()))}), 400

    file = request.files['image']

    print(f"Uploaded filename: {file.filename}")
    print(f"Uploaded content type: {file.content_type}")
    print(f"Uploaded file content length: {request.content_length}")
    print("=================")

    if file.filename == '':
        print("Validation Failure: filename is empty")
        return jsonify({'error': 'No selected image file.'}), 400

    try:
        # Basic validation for image formats
        if file.content_type and not file.content_type.startswith('image/'):
            print(f"Validation Failure: content type '{file.content_type}' is not image")
            return jsonify({'error': f"Invalid file type. Please upload an image. Received: {file.content_type}"}), 400

        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Perform prediction (3-class softmax)
        prediction = model.predict(processed_image)
        
        # Output is a single probability array for 3 classes: [Lung_Opacity, Normal, split_data]
        probs = prediction[0]
        
        class_names = ["Lung_Opacity", "Normal", "split_data"]
        class_index = int(np.argmax(probs))
        predicted_label = class_names[class_index]
        confidence = float(probs[class_index])
        
        return jsonify({
            'prediction': predicted_label,
            'class_index': class_index,
            'confidence': confidence,
            'probabilities': {
                'Lung_Opacity': float(probs[0]),
                'Normal': float(probs[1]),
                'split_data': float(probs[2])
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the server on all available interfaces on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
