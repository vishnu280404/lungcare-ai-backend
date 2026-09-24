import os
import io
import gc
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "densenet121_lung_opacity_final.keras"
)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    image_array = tf.keras.applications.densenet.preprocess_input(
        image_array
    )

    return image_array


@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "error": "Model not loaded on server."
        }), 500

    if "image" not in request.files:
        return jsonify({
            "error": "No image field in request."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No selected image file."
        }), 400

    try:
        image_bytes = file.read()

        processed_image = preprocess_image(image_bytes)

        # Direct inference
        prediction = model(
            processed_image,
            training=False
        ).numpy()[0]

        class_names = [
            "Lung_Opacity",
            "Normal",
            "split_data"
        ]

        class_index = int(np.argmax(prediction))
        predicted_label = class_names[class_index]
        confidence = float(prediction[class_index])

        result = {
            "prediction": predicted_label,
            "class_index": class_index,
            "confidence": confidence,
            "probabilities": {
                "Lung_Opacity": float(prediction[0]),
                "Normal": float(prediction[1]),
                "split_data": float(prediction[2])
            }
        }

        del processed_image
        del prediction
        gc.collect()

        return jsonify(result)

    except Exception as e:
        print("Prediction error:", e)

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "LungCare AI backend is running"
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )