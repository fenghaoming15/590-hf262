#-------------Load in data----------------
import os, shutil
import matplotlib.pyplot as plt

#-------------making directories----------


original_dataset_dir = '/Users/haomingfeng/Downloads/dogs-vs-cats/train'

base_dir = '/Users/haomingfeng/590-hf262/HW4.0/DOGS-AND-CATS/cats_and_dogs_small'
#os.mkdir(base_dir)

train_dir = os.path.join(base_dir,'train')
#os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
#os.mkdir(validation_dir)

test_dir = os.path.join(base_dir,'test')
#os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
#os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
#os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
#os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
#os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
#os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
#os.mkdir(test_dogs_dir)'''

#Load in data for the first 1000 cat into training sample
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

#load in data for the 1000-1500 cat into validation sample
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

#Load in data for 1500-200 cat into test sample
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

#Load in first 1000 dog images into training sample
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

#Load in 1000-1500 dog images into validation sample
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

#Load in 1500-2000 dog images into testing sample
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)


#----------------small conv net--------------------
from keras import layers
from keras import models
from keras import optimizers

#Create a function for the model
def cnn_model():
    #initialize the model
    model = models.Sequential()
    #Add a convnet for 3*3 and 32 of them, activation = relu, input image shape 150*150 * 3 color
    model.add(layers.Conv2D(32,(3,3),activation = 'relu',input_shape = (150,150,3)))
    #Add a max_pooling layer
    model.add(layers.MaxPooling2D((2,2)))
    #Add a convnet for 3*3 and 64 of them, activation = relu
    model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
    #Add a max_pooling layer
    model.add(layers.MaxPooling2D((2,2)))
    #Add a convnet for 3*3 and 128 of them, activation = relu
    model.add(layers.Conv2D(128,(3,3),activation = 'relu'))
    #Add a max_pooling layer
    model.add(layers.MaxPooling2D((2,2)))
    #Add a convnet for 3*3 and 128 of them, activation = relu
    model.add(layers.Conv2D(128,(3,3),activation = 'relu'))
    #Add a max_pooling layer
    model.add(layers.MaxPooling2D((2,2)))
    #Flatten our convnet
    model.add(layers.Flatten())
    #Add a DFF network of 512 neuron
    model.add(layers.Dense(512, activation='relu'))
    #Output layer with 1 output node and sigmoid activation
    model.add(layers.Dense(1, activation='sigmoid'))
    #Display the summary of our CNN network
    model.summary()
    #Compile our model
    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
    return model


#---------------Preprocess Data-------------------

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

#define a generator function
def generator():
    #Rescale image for 1/255
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    #define a generator for train and validation datasets
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size = (150,150),
                                                    batch_size = 20,
                                                    class_mode = 'binary')
    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                    target_size = (150,150),
                                                    batch_size = 20,
                                                    class_mode = 'binary')
    return train_generator, validation_generator

#Define a fit function
def fit(): 
    #fit function
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=100,
                                  epochs=30,
                                  validation_data=validation_generator,
                                  validation_steps=50)
    return history

#Define a data augmentation function and plot the graphs
def data_augmentation(train_cats_dir):
    #data augmentation function
    datagen = ImageDataGenerator(rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest')
    #directories for images
    fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
    #Randomly choose a image
    img_path = fnames[3]
    #rescale image to 150 * 150
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    #display four augmented image
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:
            break
    plt.show()

#Define a cnn network with dropout
def cnn_with_dropout():
    #Everything else is the same except for the dropout layer
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    #Here is the key a drop out hyperparameter before our DFF network
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
    return model

def fit_aug_drop():
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale = 1./255)
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size = (150,150),
                                                        batch_size = 32,
                                                        class_mode = 'binary'
                                                        )
    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size = (150,150),
                                                        batch_size = 32,
                                                        class_mode = 'binary'
                                                        )
    history = model.fit_generator(train_generator,
                                epochs = 100,
                                steps_per_epoch= 30,
                                validation_data = validation_generator,
                                validation_steps = 50)
    return history



#----------------visualize parameters, uncomment if needed-------------------
'''acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()'''