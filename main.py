#Put code below if u don't want tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow_hub as hub

#pip install tensorflow_hub 

train_path = ".\\data\\train"
test_path = ".\\data\\test"
img_size = 224
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

url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

download_model = hub.KerasLayer(url,input_shape=(img_size,img_size,3))

model = Sequential([
    download_model,
    Dense(2),
    Activation("softmax")
])

model.compile(optimizer=Adam(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# print("\n Model summary: ")
# print(model.summary())

# print("\n Model Training: ")
# model.fit(train_dataset,
#           batch_size=batch_size,
#           epochs=epochs)
        
# print("\n Model Evaluation: ")
# model.evaluate(test_dataset)

# print("\n Model save: ")
# model.save("model.h5")

load_model = tf.keras.models.load_model("model.h5",custom_objects={"KerasLayer":hub.KerasLayer})
load_model.evaluate(test_dataset)