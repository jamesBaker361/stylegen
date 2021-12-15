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

def get_encoder(inputs,input_dim,name,flat_latent,residual,attention,m=3):
    #inputs = tk.Input(shape=input_dim)
    x = layers.Conv2D(max(8,input_dim[-1]), (1, 1), (1, 1))(inputs)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(.2)(x)
    x=layers.LeakyReLU()(x)
    if residual==True:
        x = ResNextBlock(kernel_size=(4, 4))(x)
    x = layers.Conv2D(32, (1, 1), (1, 1))(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(.2)(x)
    x=layers.LeakyReLU()(x)
    for _ in range(m):
        channels = x.shape[-1] *2
        if residual==True:
            x = ResNextBlock(kernel_size=(4, 4))(x)
        x = layers.Conv2D(channels, (4, 4), (2, 2), padding='same')(x)
        x=layers.BatchNormalization()(x)
        x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        if residual==True:
            x = ResNextBlock(kernel_size=(4, 4))(x)
        x=layers.BatchNormalization()(x)
        x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        x=layers.Conv2D(channels,(3,3),padding='same')(x)
        x=layers.BatchNormalization()(x)
        x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
    if residual==True:
        x = ResNextBlock(kernel_size=(4, 4))(x)
    if attention==True:
        x = attn_block(x)
    if residual==True:
        x = ResNextBlock(kernel_size=(4, 4))(x)
    x = GroupNormalization()(x)
    x = tk.activations.swish(x)
    if flat_latent==True:
        x=layers.Flatten()(x)
        x=layers.Dense(flat_latent_dim)(x)
    return x

def make_decoder(input_dim,name,flat_latent,residual,attention):
    inputs = tk.Input(shape=input_dim,name='decoder_input')
    if flat_latent==False:
        x=inputs
        if residual==True:
            x = ResNextBlock(kernel_size=(4, 4))(x)
        if attention==True:
            x = attn_block(x)
    else:
        new_shape=(4,4, flat_latent_dim//16)
        x=layers.Reshape(new_shape,name='decoder_input_')(inputs)
    if residual==True:
        x = ResNextBlock(kernel_size=(4, 4))(x)
    while x.shape[-2]<256:
        channels = max(x.shape[-1]//2,32)
        if residual==True:
            x = ResNextBlock(kernel_size=(4, 4))(x)
        x = layers.Conv2DTranspose(channels, (4, 4), (2, 2),padding='same')(x)
        x=layers.BatchNormalization()(x)
        x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        if residual==True:
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
    return tk.Model(inputs=inputs,outputs=x,name='decoder')

def full_autoencoder(inputs,block,flat_latent,residual,attention,):
    input_shape=input_shape_dict[block]
    #inputs = tk.Input(shape=input_shape)
    x=get_encoder(inputs,input_shape,'encoder'.format(block),flat_latent,residual,attention)
    #x=get_decoder(x,x.shape[1:],'decoder'.format(block),flat_latent)
    dec=make_decoder(x.shape[1:],'decoder'.format(block),flat_latent,residual,attention)
    #x = enc(inputs)
    x=dec(x)
    return x

def aegen(block,flat_latent=False,residual=True,attention=True,output_blocks=[]):
    '''makes autoencoder based generator
    '''
    input_shape=input_shape_dict[block]
    inputs = tk.Input(shape=input_shape)
    x=full_autoencoder(inputs,block,flat_latent,residual,attention)
    x=tk.applications.vgg19.preprocess_input(x)
    if output_blocks==[]:
        output_blocks.append(block)
    vgg=vgg_layers(output_blocks)
    x=vgg(x)
    return Model(inputs=inputs, outputs=x,name='aegen')

def extract_generator(model,block,output_blocks):
    """gets the decoder part out of the generator and adds the vgg stuff

    Args:
        model: the aegen
        output_blocks: [] the list of output blocks we care about
    """
    decoder=model.get_layer('decoder')
    inputs=tk.Input(shape=decoder.input_shape[1:])
    x=decoder(inputs)
    x=tk.applications.vgg19.preprocess_input(x)
    if output_blocks==[]:
        output_blocks.append(block)
    vgg=vgg_layers(output_blocks)
    x=vgg(x)
    return Model(inputs=inputs, outputs=x,name='generator')

if __name__=='__main__':
    for block in input_shape_dict.keys():
        output_blocks=[]
        model=aegen(block,output_blocks=output_blocks)
        print(block, model.output_shape)
        gen=extract_generator(model,block,output_blocks)
