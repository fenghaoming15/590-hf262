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

x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)


#train our AE model
input_size = 784
hidden_size = 128
code_size = 32

input_img = Input(shape = (input_size,))
hidden_1 = Dense(hidden_size,activation = 'relu')(input_img)
code = Dense(code_size,activation = 'relu')(hidden_1)
hidden_2 = Dense(hidden_size,activation = 'relu')(code)
output_img = Dense(input_size,activation = 'linear')(hidden_2)

autoencoder = Model(input_img, output_img)
autoencoder.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
autoencoder.summary()
history = autoencoder.fit(x_train,x_train,epochs = 10,validation_data = (x_test,x_test))


#ploting loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss) + 1)
plt.plot(epochs,loss,'bo',label = 'Training Loss')
plt.plot(epochs, val_loss,"b",label = "Validation Loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig('HW6.1_history.png')
plt.show()


#Apply our model to mnist_fashion
from tensorflow.keras.datasets import fashion_mnist
(x_train_f, y_train_f),(x_test_f, y_test_f) = fashion_mnist.load_data()

x_train_f = x_train.astype("float32") / 255.
x_test_f = x_test.astype("float32") / 255.

x_train_f = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test_f = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))

#Define a counter to count how many anomaly are there in mnist-fashion dataset
anomaly_counter = 0

#Define our threshold as 4 times our model traning error
threshold =  4 * autoencoder.evaluate(x_train,x_train)

#create a for-loop to look at every single image in mnist-fashion and evaluate if it is anomaly
predicteds = autoencoder.predict(x_train_f)
for i in range(x_train_f.shape[0]):
    diff = x_train_f[i] - predicteds[i]
    curr_se = np.mean(diff * diff)
    if curr_se > threshold:
        anomaly_counter += 1

print("total anomaly: ", anomaly_counter)
print("anomaly percentage: ", anomaly_counter/x_train_f.shape[0] * 100)







