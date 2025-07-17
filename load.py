# Load interpreter
interpreter = tf.lite.Interpreter(model_path="recycle_model.tflite")
interpreter.allocate_tensors()

# Check input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]['shape'])

# Example test image preprocessing (Assume img is a 128x128x3 NumPy array)
# import cv2
# img = cv2.imread('sample.jpg')
# img = cv2.resize(img, (128, 128))
# img = np.expand_dims(img / 255.0, axis=0).astype(np.float32)

# interpreter.set_tensor(input_details[0]['index'], img)
# interpreter.invoke()
# output = interpreter.get_tensor(output_details[0]['index'])
# print("Prediction:", output)
