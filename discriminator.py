import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import layers,Model
import string

from group_norm import GroupNormalization
from resnext import ResNextBlock

from data_processing import vgg_layers

from other_globals import *

input_shape=input_shape_dict[block1_conv1]

def conv_discrim(block):
    input_shape=input_shape_dict[block]
    inputs=layers.Input(shape=input_shape)

    conv1_dim=max(128, input_shape[-1] //2)
    x= layers.Conv2D(conv1_dim,(4,4),(2,2),padding='same')(inputs)
    x=layers.BatchNormalization()(x)
    x=layers.LeakyReLU()(x)

    for _ in range(3):
        x = ResNextBlock(kernel_size=(4, 4))(x)
        x= layers.Conv2D(x.shape[-1] //2,(4,4),(2,2),padding='same')(x)
        x=layers.BatchNormalization()(x)
        x=layers.LeakyReLU()(x)
        x = ResNextBlock(kernel_size=(4, 4))(x)
        x=layers.BatchNormalization()(x)
        x=layers.LeakyReLU()(x)

    x=layers.Flatten()(x)
    x = layers.Dense(8)(x)
    x = layers.Dense(1,activation='sigmoid')(x)

    return tk.Model(inputs=inputs, outputs=x)

if __name__ =='__main__':
    model=conv_discrim(block1_conv1)
    model.summary()