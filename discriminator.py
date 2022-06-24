import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import layers

from group_norm import GroupNormalization
from resnext import ResNextBlock
from keras.constraints import Constraint
from numpy import log2

from dc_components import *
from generator import *

from other_globals import *
from keras import backend
# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}


def conv_discrim(block,labels=0,wasserstein=False,gp=False,max_channels=1024):
    input_shape=input_shape_dict[block]
    inputs=layers.Input(shape=input_shape)

    length=input_shape[-2]
    depth=int(log2(length))-1

    if wasserstein:
        constraint=ClipConstraint(0.01)
    else:
        constraint=Constraint()

    x=inputs
    for _ in range(depth):
        #x = ResNextBlock(kernel_size=(4, 4))(x)
        x= layers.Conv2D(min(x.shape[-1] *2,max_channels),(5,5),(2,2),padding='same',kernel_constraint=constraint,kernel_initializer=w_init)(x)
        #x=layers.BatchNormalization()(x)
        x=layers.LeakyReLU()(x)
        x=layers.Dropout(.2)(x)
        #x = ResNextBlock(kernel_size=(4, 4))(x)
        #x=layers.BatchNormalization()(x)
        #x=layers.LeakyReLU()(x)

    x=layers.Flatten()(x)
    z = x
    #z=layers.BatchNormalization()(z)
    z=layers.LeakyReLU()(z)
    if wasserstein or gp:
        z=layers.Dense(1,activation="linear")(z)
    else:
        z = layers.Dense(1,activation='sigmoid')(z)

    if labels>0:#adds classification head
        y=layers.Dropout(.2)(x)
        y=layers.Dense(labels*4)(y)
        y=layers.BatchNormalization()(y)
        y=layers.Dropout(.2)(y)
        y=layers.Dense(labels*2)(y)
        y=layers.BatchNormalization()(y)
        y=layers.Dense(labels,activation='softmax')(y)
    else:
        y=z

    return tk.Model(inputs=inputs, outputs=[z,y])

def dc_discriminator(block):
    input_shape=input_shape_dict[block]
    f = [2**i for i in range(4)]
    image_input = layers.Input(shape=input_shape)
    x = image_input
    filters = 64
    output_strides = 16
    h_output = IMG_H // output_strides
    w_output = IMG_W // output_strides

    for i in range(0, 4):
        x = conv_block(x, num_filters=f[i] * filters, kernel_size=5, strides=2)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return Model(image_input, x, name="discriminator")

if __name__ =='__main__':
    model=dc_discriminator(block1_conv1)
    model.summary()
    print(model(tf.random.normal([1, * input_shape_dict[block1_conv1]])))