import numpy as np
import pandas as pd
import cv2 as cv
import os

from PIL import Image
from io import BytesIO

from collections import Counter
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dropout, Dense
from keras import backend as K


model = Sequential()
model.add(Conv2D(32, 3, padding = 'same', activation = 'relu', input_shape =(96, 96, 3), kernel_initializer = 'he_normal'))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(5, activation = 'softmax'))


model.load_weights('best_model.hdf5')

def imageConvert(img):
    with BytesIO() as output:
        img.convert('RGB').save(output, 'BMP') 
        return output.getvalue()[14:]

def modelpredict(img):
    image = img.convert('RGB')
    np_array = np.array(img)
    cv_image = cv.cvtColor(np_array, cv.COLOR_RGB2BGR)
    cv_image = cv.resize(cv_image, (96,96))
    cv_image = cv_image.reshape(-1, 96, 96, 3)/255.0
    pred = model.predict(cv_image)
    pred_class = np.argmax(pred)
    classes = ['Mewtwo', 'Pikachu', 'Charmander', 'Bulbasaur', 'Squirtle']
    return classes[pred_class]