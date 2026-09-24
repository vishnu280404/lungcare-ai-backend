import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# TFLite model path
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "densenet121_lung_opacity.tflite"
)

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite model loaded successfully.")

except Exception as e:
    print(f"Error loading TFLite model: {e}")
    interpreter = None
    input_details = None
    output_details = None


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)

    # DenseNet121 preprocessing
    image_array = tf.keras.applications.densenet.preprocess_input(
        image_array
    )

    return image_array


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "LungCare AI backend is running",
        "model": "DenseNet121 TFLite"
    })


@app.route("/predict", methods=["POST"])
def predict():

    if interpreter is None:
        return jsonify({
            "error": "TFLite model not loaded on server."
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

        # Send image to TFLite model
        interpreter.set_tensor(
            input_details[0]["index"],
            processed_image
        )

        # Run inference
        interpreter.invoke()

        # Get prediction
        prediction = interpreter.get_tensor(
            output_details[0]["index"]
        )

        probs = prediction[0]

        class_names = [
            "Lung_Opacity",
            "Normal",
            "split_data"
        ]

        class_index = int(np.argmax(probs))
        predicted_label = class_names[class_index]
        confidence = float(probs[class_index])

        print("Prediction:", predicted_label)
        print("Confidence:", confidence)

        return jsonify({
            "prediction": predicted_label,
            "class_index": class_index,
            "confidence": confidence,
            "probabilities": {
                "Lung_Opacity": float(probs[0]),
                "Normal": float(probs[1]),
                "split_data": float(probs[2])
            }
        })

    except Exception as e:
        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )