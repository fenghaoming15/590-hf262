import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from   scipy.optimize import curve_fit
import json

# CODE PARAM
iplot=True

#-------------------------------------------
# HYPERPARAMETERS
#-------------------------------------------

# optimizer="rmsprop"
# loss_function="MeanSquaredError" 
loss_function="MeanAbsoluteError" 
learning_rate=0.1
numbers_epochs=100


#-------------------------------------------
#READ FILE AND SELECT AND PARTITION DATA
#-------------------------------------------

with open('../../DATA/weight.json') as f:
    data = json.load(f)

model_type="linear"       #weight vs age for children
# model_type="logistic"     #weight vs age for children+adults

if(model_type=="linear"):
    x=np.array(data['x']); y=np.array(data['y'])
    #ISOLATE NON ADULT DATA
    y=y[(x<18)]
    x=x[(x<18)]
    xlabel=data["xlabel"]    #age 
    ylabel=data["ylabel"]    #weight
    #NORMALIZE
    x=(x-np.mean(x))/np.std(x) 
    y=(y-np.mean(y))/np.std(y) 

#SELECT DIFFERENT DATA FOR LOGISTIC CASE
if(model_type=="logistic"):
    x=np.array(data['y']); y=np.array(data['is_adult'])
    xlabel=data["ylabel"]    #weight
    ylabel="ADULT=1 CHILD=0"
    #NORMALIZE
    x=(x-np.mean(x))/np.std(x) 

#RESHAPE (VECTOR --> Nx1 MATRIX)
x=x.reshape(len(x),1); y=y.reshape(len(y),1)

#PARTITION DATA
indices = np.random.permutation(x.shape[0])
CUT=int(0.8*x.shape[0]); #print(CUT,x.shape,indices.shape)
training_idx, val_idx = indices[:CUT], indices[CUT:]
x_train, y_train =  x[training_idx,:], y[training_idx,:]
x_val,   y_val   =  x[val_idx,:], y[val_idx,:]

#PLOT INITIAL DATA
if(iplot):
    fig, ax = plt.subplots()
    FS=18   #FONT SIZE
    ax.plot(x_train,y_train,'o')
    ax.plot(x_val,y_val,'o')
    plt.xlabel(xlabel, fontsize=FS)
    plt.ylabel(ylabel, fontsize=FS)
    plt.show()


#-------------------------------------------
#TRAIN WITH KERAS
#-------------------------------------------

# batch_size=1                      #stocastic training
# batch_size=int(len(x_train)/2.)     #batch training
batch_size=len(x_train)             #batch training

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
# from tensorflow.keras import activations

#LOGISTIC REGRESSION MODEL
if(model_type=="logistic"):
    act='sigmoid'
    model = keras.Sequential([
    layers.Dense(1,
    activation=act,
    kernel_initializer=initializers.RandomNormal(stddev=1),
    bias_initializer=initializers.RandomUniform(minval=-1, maxval=-0.5, seed=None),
    input_shape=(1,)),
    ])

if(model_type=="linear"):
    model = keras.Sequential([
    layers.Dense(1, activation='linear', input_shape=(1,)),
    ])


# LINEAR REGRESSION MODEL
print(model.summary()); #print(x_train.shape,y_train.shape)
print("initial parameters:", model.get_weights())

# COMPILING THE MODEL 
opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
model.compile(optimizer=opt,
              loss=loss_function)

# TRAINING YOUR MODEL
history = model.fit(x_train,
                    y_train,
                    epochs=numbers_epochs,
                    batch_size=batch_size,
                    validation_data=(x_val, y_val))

history_dict = history.history


#GET MODEL PARAMETERS
weights = model.get_weights() # Getting params
m=weights[0][0][0]
b=weights[1][0]
print(m,b,weights)


if(iplot):
    fig, ax = plt.subplots()
    FS=18   #FONT SIZE

    # ax.plot(model.predict(x_train),y_train,'o')
    ax.plot(x_train,y_train,'o')
    # ax.plot(x_train,model.predict(x_train) ,'x')
    x1=np.linspace(min(x_train),max(x_train),100)
    y1=m*x1+b
    if(model_type=="logistic"):
        y1=1./(1+np.exp(-(y1)))
    ax.plot(x1,y1,'-')
    ax.plot(x_val,y_val,'x')
    plt.xlabel(xlabel, fontsize=FS)
    plt.ylabel(ylabel, fontsize=FS)
    plt.show()


    # PLOTTING THE TRAINING AND VALIDATION LOSS 
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

