import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

# TFLite model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "densenet121_lung_opacity.tflite"
)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded successfully.")
print("Input:", input_details[0]["shape"])
print("Output:", output_details[0]["shape"])


# --------------------------------------------------
# BASIC X-RAY IMAGE VALIDATION
# --------------------------------------------------
def validate_image(image):
    """
    Basic heuristic validation.
    This is NOT a medical X-ray detector.
    It only rejects obviously unsuitable images.
    """

    # Convert to RGB and grayscale
    rgb = image.convert("RGB")
    gray = rgb.convert("L")

    width, height = gray.size

    # Very small image
    if width < 200 or height < 200:
        return False, "Image resolution is too small."

    # Check aspect ratio
    ratio = width / height

    if ratio < 0.5 or ratio > 2.0:
        return False, "Invalid image shape."

    arr = np.asarray(gray).astype(np.float32)

    # Calculate grayscale statistics
    mean = np.mean(arr)
    std = np.std(arr)

    # Completely blank / almost uniform image
    if std < 18:
        return False, "Image has insufficient visual information."

    # Extremely dark image
    if mean < 20:
        return False, "Image is too dark."

    # Extremely bright image
    if mean > 245:
        return False, "Image is too bright."

    # Check color saturation.
    # Chest X-rays are normally grayscale.
    rgb_arr = np.asarray(rgb).astype(np.float32)

    channel_difference = np.mean(
        np.abs(rgb_arr[:, :, 0] - rgb_arr[:, :, 1]) +
        np.abs(rgb_arr[:, :, 1] - rgb_arr[:, :, 2])
    )

    # Strongly colored image → probably not X-ray
    if channel_difference > 35:
        return False, "Image appears to be a colored photograph."

    return True, "Image passed basic validation."


# --------------------------------------------------
# PREPROCESS
# --------------------------------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image).astype(np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    image_array = tf.keras.applications.densenet.preprocess_input(
        image_array
    )

    return image_array


# --------------------------------------------------
# HOME
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "LungCare AI backend is running",
        "model": "DenseNet121 TFLite"
    })


# --------------------------------------------------
# PREDICT
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "prediction": "Invalid Image",
            "confidence": 0.0,
            "message": "Please upload an image."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "prediction": "Invalid Image",
            "confidence": 0.0,
            "message": "No image selected."
        }), 400

    try:

        # Read image
        image_bytes = file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        print("Uploaded image:", file.filename)
        print("Image size:", image.size)

        # --------------------------------------------------
        # VALIDATION BEFORE MODEL
        # --------------------------------------------------
        valid, reason = validate_image(image)

        if not valid:

            print("IMAGE REJECTED:", reason)

            return jsonify({
                "prediction": "Invalid Image",
                "confidence": 0.0,
                "message": "Please upload a valid chest X-ray image.",
                "reason": reason
            }), 400

        print("Image passed basic validation.")

        # --------------------------------------------------
        # PREPROCESS
        # --------------------------------------------------
        processed_image = preprocess_image(image)

        # --------------------------------------------------
        # TFLITE PREDICTION
        # --------------------------------------------------
        interpreter.set_tensor(
            input_details[0]["index"],
            processed_image
        )

        interpreter.invoke()

        prediction = interpreter.get_tensor(
            output_details[0]["index"]
        )

        probs = prediction[0]

        # Current model classes
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
            "prediction": "Error",
            "confidence": 0.0,
            "message": str(e)
        }), 500


# --------------------------------------------------
# LOCAL SERVER
# --------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )