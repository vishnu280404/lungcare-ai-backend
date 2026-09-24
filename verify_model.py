import os
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'densenet121_lung_opacity_final.keras')

print(f"Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    
    print("\n--- Model Architecture ---")
    print(f"Input Shape: {model.input_shape}")
    print(f"Output Shape: {model.output_shape}")
    
    print("\n--- Preprocessing information ---")
    print("Model is DenseNet121. The preprocessing used in app.py scales pixel values to [0, 1]. DenseNet typically uses [0, 1] or specific channel-wise normalization depending on how it was trained (e.g. ImageNet weights might need tf.keras.applications.densenet.preprocess_input). We'll assume the simple [0, 1] is correct if the model was trained that way, but let's confirm the shapes.")
    
except Exception as e:
    print(f"Error loading model: {e}")
