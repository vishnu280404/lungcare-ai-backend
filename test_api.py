import requests
import time
import subprocess
import os
import sys
from PIL import Image
import numpy as np

# Create a test image
img_path = 'test_image.jpg'
# Generate a dummy noise image resembling an X-ray (grayscale-ish)
noise = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(noise)
img.save(img_path)

print("Starting Flask server...")
# Start Flask server
env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'
proc = subprocess.Popen([sys.executable, 'app.py'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for server to start
time.sleep(25)

print("Sending POST request to /predict...")
url = 'http://127.0.0.1:5000/predict'
try:
    with open(img_path, 'rb') as f:
        response = requests.post(url, files={'image': f})
    
    print(f"Status Code: {response.status_code}")
    try:
        print("Response JSON:")
        print(response.json())
    except:
        print("Response Text:")
        print(response.text)
except Exception as e:
    print(f"Error during request: {e}")

# Clean up
print("Shutting down Flask server...")
proc.terminate()
try:
    proc.wait(timeout=3)
except:
    proc.kill()
    
if os.path.exists(img_path):
    os.remove(img_path)
