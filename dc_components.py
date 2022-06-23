import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.layers  import GaussianNoise

from other_globals import *

from other_globals import *
from generator import *

w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

#https://github.com/nikhilroxtomar/DCGAN-on-Anime-Faces/blob/master/gan.py

def deconv_block(inputs, num_filters, kernel_size, strides, bn=True):
    x = Conv2DTranspose(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=w_init,
        padding="same",
        strides=strides,
        use_bias=False
        )(inputs)

    if bn:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
    return x


def conv_block(inputs, num_filters, kernel_size, padding="same", strides=2, activation=True):
    x = Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=w_init,
        padding=padding,
        strides=strides,
    )(inputs)

    if activation:
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
    return x