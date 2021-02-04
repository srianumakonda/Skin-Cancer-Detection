#Put code below if u don't want tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

train_path = ".\\data\\train"
test_path = ".\\data\\test"
img_size = 128
batch_size = 32
epochs = 10

augment_train_data = ImageDataGenerator(horizontal_flip=True,
                                        rotation_range=50,
                                        zoom_range=0.2,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        rescale=1./255,
                                        validation_split=0.2)
augment_test_set = ImageDataGenerator(rescale=1./255)

train_dataset = augment_train_data.flow_from_directory(train_path,
                                                       shuffle=True,
                                                       target_size=(img_size,img_size),
                                                       batch_size=batch_size)
test_dataset = augment_train_data.flow_from_directory(test_path,
                                                       target_size=(img_size,img_size),
                                                       batch_size=batch_size)

fig = plt.figure(figsize=(15,10))

# for i in range(1,10):
#     plt.subplot(3,3,i)
#     plt.imshow(train_dataset[0][0][i-1])
# plt.show()


model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(img_size,img_size,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dense(2))
model.add(Activation("softmax"))

model.compile(optimizer=Adam(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_dataset,
          batch_size=batch_size,
          epochs=epochs)