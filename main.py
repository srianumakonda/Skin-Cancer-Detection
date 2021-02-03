import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = ".\\data\\train"
test_path = ".\\data\\test"
img_size = 224
batch_size = 32

augment_train_data = ImageDataGenerator(horizontal_flip=True,
                                        rotation_range=50,
                                        zoom_range=0.2,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        rescale=1./255,
                                        validation_split=0.1)
augment_test_set = ImageDataGenerator(rescale=1./255)

train_dataset = augment_train_data.flow_from_directory(train_path,
                                                       shuffle=True,
                                                       target_size=(img_size,img_size),
                                                       batch_size=batch_size)
test_dataset = augment_train_data.flow_from_directory(test_path,
                                                       target_size=(img_size,img_size),
                                                       batch_size=batch_size)

fig = plt.figure(figsize=(15,10))

for i in range(1,10):
    plt.subplot(3,3,i)
    plt.imshow(train_dataset[0][0][i-1])
plt.show()