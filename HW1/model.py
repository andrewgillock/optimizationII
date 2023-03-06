import numpy as np
import pandas as pd
import tensorflow as tf

def image_classifier(file):

    # first determine if grayscale values are normalized
    if file.max().values.mean() > 1:
        file = file/255

    # reshape to make tf happy
    image = file.to_numpy().reshape(1, 28, 28, 1)

    # load saved model
    model = tf.keras.models.load_model('my_model')

    # get probabilities of each number, choose index of highest one
    predicted = model.predict(image).argmax()
    
    return predicted
