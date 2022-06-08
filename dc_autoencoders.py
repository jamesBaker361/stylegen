import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import layers,Model
from tensorflow.keras.layers import *
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers  import GaussianNoise

from tensorflow.python.keras.activations import sigmoid

from group_norm import GroupNormalization
from resnext import ResNextBlock

from other_globals import *

from data_processing import vgg_layers
from rescaling import Rescaling

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

def dc_encoder(input_shape,latent_dim):
    inputs = tk.Input(shape=input_shape)
    if input_shape[-1]<64:
        x=conv_block(inputs,64,kernel_size=(1,1),strides=(1,1))
    else:
        x=inputs
    while x.shape[-2]>16:
        channels= min(x.shape[-1]*2,512)
        x=conv_block(x,channels,(5,5),strides=(4,4))
    while x.shape[-2]>2:
        channels= min(x.shape[-1]*2,512)
        x=conv_block(x,channels,(3,3),strides=(2,2))
    x=Flatten()(x)
    x=Dense(latent_dim)(x)

    return Model(inputs,x,name="encoder")


def dc_decoder(latent_dim,filters = 32,output_strides = 16):
    f = [2**i for i in range(5)][::-1]
    h_output = image_dim[0] // output_strides
    w_output = image_dim[1] // output_strides

    noise = Input(shape=(latent_dim,), name="generator_noise_input")

    x = Dense(f[0] * filters * h_output * w_output, use_bias=False)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((h_output, w_output, 16 * filters))(x)

    for i in range(1, 5):
        x = deconv_block(x,
            num_filters=f[i] * filters,
            kernel_size=5,
            strides=2,
            bn=True
        )

    x = conv_block(x,
        num_filters=3,  ## Change this to 1 for grayscale.
        kernel_size=5,
        strides=1,
        activation=False
    )
    x = Activation("tanh")(x)
    x=Rescaling(255,offset=127.5,name='img_output')(x)

    return Model(noise, x, name="decoder")

if __name__ == "__main__":
    dec=dc_decoder(1024)
    dec.summary()