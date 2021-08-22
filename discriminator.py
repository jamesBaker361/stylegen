import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import layers,Model
import string

from group_norm import GroupNormalization
from resnext import ResNextBlock

from data_processing import vgg_layers

from other_globals import *

input_shape=input_shape_dict[block1_conv1]

def conv_discrim(input_shape=input_shape):
    inputs=layers.Input(shape=input_shape)

    x= layers.Conv2D(input_shape[-1] //2,(4,4),(2,2),padding='same')(inputs)
    x=layers.BatchNormalization()(x)
    x=layers.LeakyReLU()(x)

    for _ in range(3):
        x= layers.Conv2D(x.shape[-1] //2,(4,4),(2,2),padding='same')(x)
        x=layers.BatchNormalization()(x)
        x=layers.LeakyReLU()(x)

    x=layers.Flatten()(x)
    x = layers.Dense(1,activation='sigmoid')(x)

    return tk.Model(inputs=inputs, outputs=x)

if __name__ =='__main__':
    model=conv_discrim()
    model.summary()