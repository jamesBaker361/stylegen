import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import layers,Model
import string

from tensorflow.python.keras.activations import sigmoid

from group_norm import GroupNormalization
from resnext import ResNextBlock

from data_processing import vgg_layers
from rescaling import Rescaling

from other_globals import *
from generator import *

flat_latent_dim=64 #the dim of latent space is dim is 1-D

def get_encoder(input_dim,name,flat_latent,m=3):
    inputs = tk.Input(shape=input_dim)
    x = layers.Conv2D(max(8,input_dim[-1]), (1, 1), (1, 1))(inputs)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(.2)(x)
    x=layers.LeakyReLU()(x)
    x = ResNextBlock(kernel_size=(4, 4))(x)
    x = layers.Conv2D(32, (1, 1), (1, 1))(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(.2)(x)
    x=layers.LeakyReLU()(x)
    for _ in range(m):
        channels = x.shape[-1] *2
        x = ResNextBlock(kernel_size=(4, 4))(x)
        x = layers.Conv2D(channels, (4, 4), (2, 2), padding='same')(x)
        x=layers.BatchNormalization()(x)
        x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        x = ResNextBlock(kernel_size=(4, 4))(x)
        x=layers.BatchNormalization()(x)
        x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        x=layers.Conv2D(channels,(3,3),padding='same')(x)
        x=layers.BatchNormalization()(x)
        x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
    x = ResNextBlock(kernel_size=(4, 4))(x)
    x = attn_block(x)
    x = ResNextBlock(kernel_size=(4, 4))(x)
    x = GroupNormalization()(x)
    x = tk.activations.swish(x)
    if flat_latent==True:
        x=layers.Flatten()(x)
        x=layers.Dense(flat_latent_dim)(x)
    return Model(inputs=inputs, outputs=x,name=name)

def get_decoder(input_dim,name,flat_latent):
    inputs = tk.Input(shape=input_dim,name='decoder_input')
    if flat_latent==False:
        x=inputs
        x = ResNextBlock(kernel_size=(4, 4))(x)
        x = attn_block(x)
    else:
        new_shape=(4,4, flat_latent_dim//16)
        x=layers.Reshape(new_shape)(inputs)
    x = ResNextBlock(kernel_size=(4, 4))(x)
    while x.shape[-2]<256:
        channels = max(x.shape[-1]//2,32)
        x = ResNextBlock(kernel_size=(4, 4))(x)
        x = layers.Conv2DTranspose(channels, (4, 4), (2, 2),padding='same')(x)
        x=layers.BatchNormalization()(x)
        x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        x = ResNextBlock(kernel_size=(4, 4))(x)
        x=layers.BatchNormalization()(x)
        x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        x=layers.Conv2D(channels,(3,3),padding='same')(x)
        x=layers.BatchNormalization()(x)
        x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
    x = GroupNormalization(groups=x.shape[-1] // 4)(x)
    x = tk.activations.swish(x)
    x = layers.Conv2D(3, (1, 1), (1, 1))(x)
    x=layers.Activation('sigmoid')(x)
    x=Rescaling(255,name='img_output')(x)
    return Model(inputs=inputs, outputs=x,name=name)

def full_autoencoder(block,flat_latent):
    input_shape=input_shape_dict[block]
    inputs = tk.Input(shape=input_shape)
    enc=get_encoder(input_shape,'encoder'.format(block),flat_latent)
    dec=get_decoder(enc.output_shape[1:],'decoder'.format(block),flat_latent)
    
    x = enc(inputs)
    x=dec(x)
    return Model(inputs=inputs, outputs=x,name='autoencoder'.format(block))

def aegen(block,flat_latent=False,output_blocks=[]):
    '''makes autoencoder based generator
    '''
    input_shape=input_shape_dict[block]
    inputs = tk.Input(shape=input_shape)
    autoencoder=full_autoencoder(block,flat_latent)
    x=autoencoder(inputs)
    x=tk.applications.vgg19.preprocess_input(x)
    output_blocks.append(block)
    vgg=vgg_layers(output_blocks)
    x=vgg(x)
    return Model(inputs=inputs, outputs=x,name='aegen')

if __name__=='__main__':
    for b in input_shape_dict.keys():
        model=aegen(b,output_blocks=[])
        gen_inputs=model.get_layer('autoencoder').get_layer('decoder')
        gen_outputs=model.outputs
        generator=tk.Sequential()
        generator.summary()