import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import layers

from group_norm import GroupNormalization
from resnext import ResNextBlock

from data_processing import vgg_layers
from generator import *

from other_globals import *

def conv_discrim(block,labels=0,attention=False):
    """[summary]

    Args:
    -----
        block ([str]): [description]
        labels (int, optional): [description]. Defaults to 0.

    Returns:
    ------
        tf.keras.Model : the discriminator; returns either value between 0-1, and if labels>0, a classification vector [0,0,0,,,1,,0]
    """
    input_shape=input_shape_dict[block]
    inputs=layers.Input(shape=input_shape)

    conv1_dim=max(128, input_shape[-1] //2)
    x= layers.Conv2D(conv1_dim,(4,4),(2,2),padding='same')(inputs)
    x=layers.BatchNormalization()(x)
    x=layers.LeakyReLU()(x)
    x=layers.Dropout(.2)(x)
    if attention==True:
        x=attn_block(x)

    for _ in range(3):
        x = ResNextBlock(kernel_size=(4, 4))(x)
        x= layers.Conv2D(x.shape[-1] //2,(4,4),(2,2),padding='same')(x)
        x=layers.BatchNormalization()(x)
        x=layers.LeakyReLU()(x)
        x=layers.Dropout(.2)(x)
        #x = ResNextBlock(kernel_size=(4, 4))(x)
        #x=layers.BatchNormalization()(x)
        #x=layers.LeakyReLU()(x)

    if attention==True:
        x=attn_block(x)
    x=layers.Flatten()(x)
    z = layers.Dense(8)(x)
    z=layers.BatchNormalization()(z)
    z=layers.LeakyReLU()(z)
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



if __name__ =='__main__':
    model=conv_discrim(block1_conv1,0)
    model.summary()
    print(model(tf.random.normal([1, * input_shape_dict[block1_conv1]])))