

import pandas  as  pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#----------------------------------------
#GET DATA
#----------------------------------------

#The Auto MPG dataset
#The dataset is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/).
#First download and import the dataset using pandas:

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

df = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)


#----------------------------------------
#VISUALIZE DATA
#----------------------------------------

#IMPORT FILE FROM CURRENT DIRECTORY
import Seaborn_visualizer as SBV


SBV.get_pd_info(df)
# SBV.pd_general_plots(df,HUE='Origin')
# SBV.pandas_2D_plots(df,col_to_plot=[1,4,5],HUE='Origin')

#----------------------------------------
#PRE-PROCESS DATA 
#(EXTRACT AND CONVERT TO TENSOR)
#----------------------------------------

print("----------------------")
print("EXTRACT DATA")
print("----------------------")

#SELECT COLUMNS TO USE AS VARIABLES 
x_col=[2]; 
# x_col=[2,4,5]; 
# x_col=[1,2,3,4,5]; 
y_col=[0];  xy_col=x_col+y_col
x_keys =SBV.index_to_keys(df,x_col)        #dependent var
y_keys =SBV.index_to_keys(df,y_col)        #independent var
xy_keys=SBV.index_to_keys(df,xy_col)        #independent var

print("X=",x_keys); print("Y=",y_keys);  #
# SBV.pd_general_plots(df[xy_keys])

#CONVERT SELECT DF TO NP
x=df[x_keys].to_numpy()
y=df[y_keys].to_numpy()

#REMOVE NAN IF PRESENT
xtmp=[]; ytmp=[];
for i in range(0,len(x)):
    if(not 'nan' in str(x[i])):
        xtmp.append(x[i])
        ytmp.append(y[i])
x=np.array(xtmp); y=np.array(ytmp)

#PARTITION DATA
fraction_train=0.8
indices = np.random.permutation(x.shape[0])
CUT=int(fraction_train*x.shape[0]); #print(CUT,x.shape,indices.shape)
training_idx, test_idx = indices[:CUT], indices[CUT:]
x_train, y_train =  x[training_idx,:], y[training_idx,:]
x_val,   y_val   =  x[test_idx,:], y[test_idx,:]

# print(x_train.shape,y_train.shape)
# print(x_val.shape,y_val.shape)

#NORMALIZE DATA
# print(np.mean(x_train,axis=0),np.std(x_train,axis=0))
x_mean=np.mean(x,axis=0); x_std=np.std(x,axis=0)
y_mean=np.mean(y,axis=0); y_std=np.std(y,axis=0)
x_train=(x_train-x_mean)/x_std
x_val=(x_val-x_mean)/x_std
y_train=(y_train-y_mean)/y_std
y_val=(y_val-y_mean)/y_std

# # #PLOT INITIAL DATA
iplot=False
if(iplot):
    fig, ax = plt.subplots()
    FS=18   #FONT SIZE

    for indx in range(0,x_train.shape[1]):
        plt.plot(x_train[:,indx],y_train,'o')
        plt.plot(x_val[:,indx],y_val,'o')
        plt.xlabel(x_keys[indx], fontsize=FS)
        plt.ylabel(y_keys[0], fontsize=FS)
        plt.show(); plt.clf()

# exit()
#-------------------------------------------
#TRAIN WITH KERAS
#-------------------------------------------

# HYPERPARAMETERS 
optimizer="rmsprop"
loss_function="MeanSquaredError" 
#loss_function="MeanAbsoluteError" 
learning_rate=0.051
numbers_epochs=200
model_type="linear"
input_shape=(x_train.shape[1],);

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
    input_shape=(1,)),
    ])

if(model_type=="linear"):
    model = keras.Sequential([
    layers.Dense(1, activation='linear', input_shape=input_shape),
    ])


# LINEAR REGRESSION MODEL
print(model.summary()); #print(x_train.shape,y_train.shape)
print("initial parameters:", model.get_weights())

# COMPILING THE MODEL 
opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
model.compile(optimizer=opt,loss=loss_function
              )

# TRAINING YOUR MODEL
history = model.fit(x_train,
                    y_train,
                    epochs=numbers_epochs,
                    batch_size=batch_size,
                    validation_data=(x_val, y_val))

history_dict = history.history

yp=model.predict(x_train)
yp_val=model.predict(x_val) 

#UN-NORMALIZE DATA (CONVERT BACK TO ORIGINAL UNITS)
x_train=x_std*x_train+x_mean 
x_val=x_std*x_val+x_mean 
y_train=y_std*y_train+y_mean 
y_val=y_std*y_val+y_mean 
yp=y_std*yp+y_mean 
yp_val=y_std*yp_val+y_mean 

# print(input_shape,x_train.shape,yp.shape,y_train.shape)

# #PLOT INITIAL DATA
iplot=True
if(iplot):
    FS=18   #FONT SIZE

    #PARITY PLOT
    plt.plot(yp,yp,'-')
    plt.plot(yp,y_train,'o')
    plt.xlabel("y (predicted)", fontsize=FS)
    plt.ylabel("y (data)", fontsize=FS)
    plt.show()
    plt.clf()

    #FEATURE DEPENDENCE
    for indx in range(0,x_train.shape[1]):
        #TRAINING
        plt.plot(x_train[:,indx],y_train,'ro')
        plt.plot(x_train[:,indx],yp,'bx')
        plt.xlabel(x_keys[indx], fontsize=FS)
        plt.ylabel(y_keys[0], fontsize=FS)
        plt.show()
        plt.clf()

        plt.plot(x_val[:,indx],y_val,'ro')
        plt.plot(x_val[:,indx],yp_val,'bx')
        plt.xlabel(x_keys[indx], fontsize=FS)
        plt.ylabel(y_keys[0], fontsize=FS)
        plt.show()
        plt.clf()

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

