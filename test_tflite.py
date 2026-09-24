import tensorflow as tf
from PIL import Image
import numpy as np

IMAGE_PATH = r"D:\lungcare AI\test_images\Lung_Opacity-1579.png"
MODEL_PATH = r"model\densenet121_lung_opacity.tflite"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize((224, 224))

image_array = np.array(image).astype("float32")
image_array = np.expand_dims(image_array, axis=0)

# DenseNet preprocessing
image_array = tf.keras.applications.densenet.preprocess_input(image_array)

# Set input
interpreter.set_tensor(input_details[0]["index"], image_array)

# Prediction
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]["index"])
probs = output[0]

class_names = ["Lung_Opacity", "Normal", "split_data"]

class_index = int(np.argmax(probs))
predicted_label = class_names[class_index]
confidence = float(probs[class_index])

print("\n===== TFLITE TEST =====")
print("Image:", IMAGE_PATH)
print("Prediction:", predicted_label)
print("Confidence:", confidence)
print("Probabilities:")
print("Lung_Opacity:", float(probs[0]))
print("Normal:", float(probs[1]))
print("split_data:", float(probs[2]))
print("=======================\n")