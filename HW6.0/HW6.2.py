import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from keras.layers import Dense, Input

#read in data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

print(x_train.shape)
print(x_test.shape)

input = layers.Input(shape = (28,28,1))



#Encoder
x = layers.Conv2D(32,(3,3), activation = 'relu', padding = "same")(input)
x = layers.MaxPooling2D((2,2), padding = 'same')(x)
x = layers.Conv2D(32,(3,3), activation = 'relu',padding = 'same')(x)
x = layers.MaxPooling2D((2,2), padding = 'same')(x)

#Decoder
x = layers.Conv2DTranspose(32,(3,3),strides = 2, activation = 'relu',padding = 'same')(x)
x = layers.Conv2DTranspose(32,(3,3),strides = 2, activation = 'relu',padding = 'same')(x)
x = layers.Conv2D(1,(3,3),activation = 'sigmoid',padding = 'same')(x)

#Autoencoder
autoencoder = Model(input,x)
autoencoder.compile(optimizer = 'rmsprop', loss = "mean_squared_error")
autoencoder.summary()

#train
history = autoencoder.fit(x_train,x_train,epochs = 10,validation_data = (x_test,x_test))


