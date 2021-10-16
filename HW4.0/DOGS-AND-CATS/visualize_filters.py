from keras.applications.vgg16 import VGG16
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#Import model
model = VGG16(weights = 'imagenet',
            include_top = False)

#Get which layer
layer_name = 'block3_conv1'
filter_index = 0

#Apply
layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:,:,:,filter_index])

#get the gradient of this loss with respect to the model's input
grads = K.gradients(loss,model.input)[0]
#normalize with l2 norm
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

iterate = K.function([model.input],[loss, grads])

import numpy as np
loss_value, grads_value = iterate([np.zeros((1,150,150,3))])

#start from a gray image with some noise
input_img_data = np.random.random((1,150,150,3)) * 20 + 128

#Magnitude of each gradient update
step = 1
#run gradient ascent for 40 steps
for i in range(40):
    loss_value, grads_value = iterate([input_img_data])

    input_img_data += grads_value * step

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x,0,1)

    x *= 255
    x = np.clip(x,0,255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size = 150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:,filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input],[loss,grads])
    input_img_data = np.random.random((1,size,size,3)) * 20 + 128.

    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)

plt.imshow(generate_pattern('block3_conv1',0))
plt.show()

#Generate a grid of all filter response patterns in a layer
layer_name = 'block1_conv1'
size = 64
margin = 5
#empty(black) image to store results
results = np.zeros((8*size + 7*margin, 8*size + 7*margin, 3))
#iterates over the rows of the results grid
for i in range(8):
    for j in range(8):
        #generates the pattern for filter i + (j * 8) in layer_name
        filter_img = generate_pattern(layer_name, i + (j*8) , size = size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start : horizontal_end, vertical_start:vertical_end, :] = filter_img

plt.figure(figsize = (20,20))
plt.imshow(results)
plt.show()

