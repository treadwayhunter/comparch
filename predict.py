import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import PIL as pillow
from PIL import Image

# Load the saved model
print()
print("predict.py")
print("tensorflow version: " + tf.__version__)
print("pillow version: " + pillow.__version__)
print("numpy version: " + np.__version__)
print("-------------------------------------")
model = keras.models.load_model('my_model.h5')

for filename in os.listdir('images'):
    img = Image.open('images/'+filename).convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.reshape(1, 28, 28)
    img = img / 255.0
    prediction = model.predict(img)
    predicted_number = np.argmax(prediction)
    sorted_indices = np.argsort(prediction[0])[::-1]

    top3_indices = sorted_indices[:3]
    top3_probabilities = prediction[0][top3_indices]

    print(filename)
    for i in range(3):
        print(f"#{i+1}: {top3_indices[i]} (probability: {top3_probabilities[i]:.4f})")
    print()