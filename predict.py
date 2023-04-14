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



cor_dict = {"right": 0, "wrong": 0}
cor_percents = []
wrong_percents = []
total = 0
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

    numWord = filename.split('_')[1]
    numWord = numWord.split('.')[0]
    # numWord is the  



    #print(numWord)
    print(filename)
    print(numWord)
    print(top3_indices[0])


    if int(top3_indices[0]) == int(numWord):
        #print('They are equal')
        cor_dict["right"] += 1
        cor_percents.append(top3_probabilities[0])
    else:
        cor_dict["wrong"] += 1
        wrong_percents.append(top3_probabilities[0])
    total += 1
    #for i in range(3):
        #print(f"#{i+1}: {top3_indices[i]} (probability: {top3_probabilities[i]:.4f})")
    #print()
#print(cor_percents)
#print(wrong_percents)

c_avg = 0
w_avg = 0
for c in cor_percents:
    c_avg += c
c_avg = c_avg/(len(cor_percents))

for w in wrong_percents:
    w_avg += w
w_avg = w_avg/(len(wrong_percents))
print("Total number of images: " + str(total))
print("Number of correct predictions: " + str(len(cor_percents)))
print("Number of wrong predictions: " + str(len(wrong_percents)))
print(f"Average confidence correct predictions: {c_avg:.4f}")
print(f"Average confidence wrong predictions: {w_avg:.4f}")



