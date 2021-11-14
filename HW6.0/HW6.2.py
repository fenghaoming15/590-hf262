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


#https://keras.io/examples/vision/autoencoder/
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


#ploting loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss) + 1)
plt.plot(epochs,loss,'bo',label = 'Training Loss')
plt.plot(epochs, val_loss,"b",label = "Validation Loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig('HW6.2_history.png')
plt.show()


#Apply our model to mnist_fashion
from tensorflow.keras.datasets import fashion_mnist
(x_train_f, y_train_f),(x_test_f, y_test_f) = fashion_mnist.load_data()

x_train_f = x_train.astype("float32") / 255.
x_test_f = x_test.astype("float32") / 255.

#Define a counter to count how many anomaly are there in mnist-fashion dataset
anomaly_counter = 0

#Define our threshold as 4 times our model traning error
threshold =  4 * autoencoder.evaluate(x_train,x_train)

#https://visualstudiomagazine.com/articles/2019/03/01/neural-anomaly-detection-using-keras.aspx
#create a for-loop to look at every single image in mnist-fashion and evaluate if it is anomaly
predicteds = autoencoder.predict(x_train_f)
for i in range(x_train_f.shape[0]):
    diff = x_train_f[i] - predicteds[i]
    curr_se = np.mean(diff * diff)
    if curr_se > threshold:
        anomaly_counter += 1

print("total anomaly: ", anomaly_counter)
print("anomaly percentage: ", anomaly_counter/x_train_f.shape[0] * 100)

