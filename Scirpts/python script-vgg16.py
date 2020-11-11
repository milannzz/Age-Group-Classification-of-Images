import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import random

#Loading data

training_data = []

path = "H:/HackerEarth/HackerEarth-Friendship goals/Friendship Dataset/dataset/Train Data/Train Data"

for images_label in os.listdir(path):
    if "x" in images_label:
        class_num = 0
    elif "y" in images_label:
        class_num = 1
    else :
        class_num = 2
    
    img = cv.imread(os.path.join(path,images_label),cv.IMREAD_GRAYSCALE)
    img = cv.resize(img,(128,128))
    training_data.append([img,class_num])

testing_data = []
path_test = "H:/HackerEarth/HackerEarth-Friendship goals/Friendship Dataset/dataset/Test Data/Test Data"
for test_img in os.listdir(path_test):
    img_test = cv.imread(os.path.join(path_test,test_img),cv.IMREAD_GRAYSCALE) 
    img_test = cv.resize(img_test,(128,128))
    testing_data.append(img_test)

testing_data = np.array(testing_data).reshape(-1,128,128,1)

random.shuffle(training_data)

X = []
y = []
for image , label in training_data:
    X.append(image)
    y.append(label)

X = np.array(X).reshape(-1,128,128,1)
y = np.array(y).reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y).toarray()

# MAking model artitecture
     
from keras.models import Sequential
from keras import layers
from keras import models
from keras import optimizers

model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  #Dropout for regularization

model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.25))

model.add(layers.Dense(256, activation='relu'))
                                                                                                                                                                                                             
model.add(layers.Dense(3, activation='softmax'))  

model.compile(loss = "categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

# Data Augmntation

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range =40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   validation_split=0.2,
                                   horizontal_flip = True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow(X,y,
                                  batch_size = 32,
                                  subset="training")

validation_set = train_datagen.flow(X,y,
                                  batch_size = 32,
                                  subset="validation")

test_set = test_datagen.flow(testing_data,
                             batch_size = 32)

history = model.fit_generator(
        training_set,
        steps_per_epoch=32,
        epochs=32,
        validation_data=validation_set,
        validation_steps=32,
        workers=8)

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#history =  model.fit(X,y,batch_size=16,epochs=16,validation_split=0.1)

y_pred = model.predict(testing_data)


        
    