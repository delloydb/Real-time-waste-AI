# Convert model to .tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open("recycle_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved successfully!")
