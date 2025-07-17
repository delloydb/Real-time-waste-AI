# Real-time-waste-AI# 🧠 Edge AI Waste Classifier

This project presents a lightweight **Edge AI solution** for real-time waste classification using **TensorFlow Lite**. It is designed to run efficiently on low-power edge devices like **Raspberry Pi**, enabling smart bins or environmental systems to detect **recyclable vs non-recyclable** waste without relying on cloud servers.

---

## 📌 Overview

- ✅ Train a compact CNN to classify images of trash.
- ✅ Convert the model to `.tflite` for edge deployment.
- ✅ Demonstrate how Edge AI improves **latency**, **privacy**, and **offline usability**.
- ✅ Ideal for smart cities, environmental tech, and IoT ecosystems.

---

## 📁 Project Structure

🛠 Tools & Libraries
TensorFlow & Keras

TensorFlow Lite

Google Colab

Python 3.9+

📊 Dataset Used
Waste Classification Dataset (Kaggle)
📎 https://www.kaggle.com/datasets/techsash/waste-classification-data

Classes used:

Recyclable: cardboard, glass, metal, paper, plastic

Non-Recyclable: trash

For this project, we reduced it to binary classification for model simplicity.

🧠 Model Summary
Layer Type	Output Shape	Params
Rescaling	(128, 128, 3)	0
Conv2D (16) + ReLU	(126, 126, 16)	448
MaxPooling2D	(63, 63, 16)	0
Conv2D (32) + ReLU	(61, 61, 32)	4,640
MaxPooling2D	(30, 30, 32)	0
Flatten	(28,800)	0
Dense (64)	(64)	1,843,264
Dropout (0.3)	-	-
Output (1, Sigmoid)	(1)	65

Loss Function: Binary Crossentropy
Optimizer: Adam
Epochs: 10
Validation Accuracy: ~89%

🧪 How to Run
🌐 Option 1: Google Colab (Recommended)
Open model_training.ipynb in Google Colab.

Upload and unzip the dataset into /content/dataset.

Run each cell sequentially.

The .tflite model will be generated at the end.

💻 Option 2: Local (with Python)

# Clone this repo
git clone https://github.com/YOUR_USERNAME/edge-ai-waste-classifier.git
cd edge-ai-waste-classifier

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook or convert to script
jupyter notebook model_training.ipynb
📦 Sample Inference (TFLite)
To load and run the .tflite model on an edge device:


import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
interpreter = tf.lite.Interpreter(model_path="recycle_model.tflite")
interpreter.allocate_tensors()

# Prepare image
img = Image.open("test.jpg").resize((128, 128))
img = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

# Set input & run inference
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
interpreter.set_tensor(input_index, img)
interpreter.invoke()

# Output
prediction = interpreter.get_tensor(output_index)
print("Recyclable" if prediction[0][0] > 0.5 else "Non-Recyclable")
💡 Why Edge AI?
Benefit	Description
⚡ Real-time	No network delays — instant classification
🔒 Privacy	No image data sent to external servers
🌍 Offline Ready	Works in remote locations
🔋 Energy-Efficient	Optimized for low-power devices

🔍 Future Improvements
Add multi-class support (e.g., detect paper vs plastic).

Deploy to Raspberry Pi or Android app.

Add voice or light feedback for smart bin integration.

Add auto-update with cloud syncing for remote insights.

🧾 License
This project is open-source under the MIT License.

🛠 Tools & Libraries
TensorFlow & Keras

TensorFlow Lite

Google Colab

Python 3.9+

📊 Dataset Used
Waste Classification Dataset (Kaggle)
📎 https://www.kaggle.com/datasets/techsash/waste-classification-data

Classes used:

Recyclable: cardboard, glass, metal, paper, plastic

Non-Recyclable: trash

For this project, we reduced it to binary classification for model simplicity.

🧠 Model Summary
Layer Type	Output Shape	Params
Rescaling	(128, 128, 3)	0
Conv2D (16) + ReLU	(126, 126, 16)	448
MaxPooling2D	(63, 63, 16)	0
Conv2D (32) + ReLU	(61, 61, 32)	4,640
MaxPooling2D	(30, 30, 32)	0
Flatten	(28,800)	0
Dense (64)	(64)	1,843,264
Dropout (0.3)	-	-
Output (1, Sigmoid)	(1)	65

Loss Function: Binary Crossentropy
Optimizer: Adam
Epochs: 10
Validation Accuracy: ~89%

🧪 How to Run
🌐 Option 1: Google Colab (Recommended)
Open model_training.ipynb in Google Colab.

Upload and unzip the dataset into /content/dataset.

Run each cell sequentially.

The .tflite model will be generated at the end.

💻 Option 2: Local (with Python)

# Clone this repo
git clone https://github.com/YOUR_USERNAME/edge-ai-waste-classifier.git
cd edge-ai-waste-classifier

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook or convert to script
jupyter notebook model_training.ipynb
📦 Sample Inference (TFLite)
To load and run the .tflite model on an edge device:


import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
interpreter = tf.lite.Interpreter(model_path="recycle_model.tflite")
interpreter.allocate_tensors()

# Prepare image
img = Image.open("test.jpg").resize((128, 128))
img = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

# Set input & run inference
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
interpreter.set_tensor(input_index, img)
interpreter.invoke()

# Output
prediction = interpreter.get_tensor(output_index)
print("Recyclable" if prediction[0][0] > 0.5 else "Non-Recyclable")
💡 Why Edge AI?
Benefit	Description
⚡ Real-time	No network delays — instant classification
🔒 Privacy	No image data sent to external servers
🌍 Offline Ready	Works in remote locations
🔋 Energy-Efficient	Optimized for low-power devices

🔍 Future Improvements
Add multi-class support (e.g., detect paper vs plastic).

Deploy to Raspberry Pi or Android app.

Add voice or light feedback for smart bin integration.

Add auto-update with cloud syncing for remote insights.

🧾 License
This project is open-source under the MIT License.

# funfact - i dont know what am doing very crazy ai code am so jogged, like whaaaaaaaaat 
