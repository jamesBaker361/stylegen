import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import layers,Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow_addons.layers import InstanceNormalization
import string

from tensorflow.python.keras.activations import sigmoid

from group_norm import GroupNormalization
from resnext import ResNextBlock

from data_processing import vgg_layers
from rescaling import Rescaling

from other_globals import *
from generator import *

 #the dim of latent space is dim is 1-D

def get_encoder(inputs,input_dim,residual,attention,m=3,base_flat_noise_dim=0,norm="instance"):
    normalization=InstanceNormalization
    if norm == "batch":
        normalization=BatchNormalization
    elif norm == "group":
        normalization=GroupNormalization
    #inputs = tk.Input(shape=input_dim)
    x = layers.Conv2D(max(8,input_dim[-1]), (1, 1), (1, 1))(inputs)
    x=normalization()(x)
    x=layers.Dropout(.2)(x)
    x=layers.LeakyReLU()(x)
    if residual==True:
        x = ResNextBlock(kernel_size=(4, 4))(x)
    x = layers.Conv2D(32, (1, 1), (1, 1))(x)
    x=normalization()(x)
    x=layers.Dropout(.2)(x)
    x=layers.LeakyReLU()(x)
    for _ in range(m):
        channels = x.shape[-1] *2
        if residual==True:
            x = ResNextBlock(kernel_size=(4, 4))(x)
        #print(x.shape)
        x = layers.Conv2D(channels, (4, 4), (2, 2), padding='same')(x)
        x=normalization()(x)
        x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        if residual==True:
            x = ResNextBlock(kernel_size=(4, 4))(x)
        x=normalization()(x)
        x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        x=layers.Conv2D(channels,(3,3),padding='same')(x)
        x=normalization()(x)
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
    if base_flat_noise_dim>0:
        x=layers.Flatten()(x)
        x=layers.Dense(base_flat_noise_dim)(x)
    return x

def make_decoder(input_dim,residual,attention,flat_latent_dim=0,norm="instance"):
    normalization=InstanceNormalization
    if norm == "batch":
        normalization=BatchNormalization
    elif norm == "group":
        normalization=GroupNormalization
    inputs = tk.Input(shape=input_dim,name='decoder_input')
    x=inputs
    if flat_latent_dim>0:
        new_shape=(2,2, 64)
        x=layers.Dense(256)(x)
        x=layers.Reshape(new_shape,name='decoder_reshape_')(x)
    if residual==True:
        x = ResNextBlock(kernel_size=(4, 4))(x)
    while x.shape[-2]<256:
        channels = max(x.shape[-1]//2,32)
        if residual==True:
            x = ResNextBlock(kernel_size=(4, 4))(x)
        x = layers.Conv2DTranspose(channels, (4, 4), (2, 2),padding='same')(x)
        x=normalization()(x)
        x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        if residual==True:
            x = ResNextBlock(kernel_size=(4, 4))(x)
        x=normalization()(x)
        x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
    x = GroupNormalization(groups=x.shape[-1] // 4)(x)
    x = tk.activations.swish(x)
    x = layers.Conv2D(3, (1, 1), (1, 1))(x)
    x=layers.Activation('sigmoid')(x)
    x=Rescaling(255,name='img_output')(x)
    return tk.Model(inputs=inputs,outputs=x,name='decoder')

def full_autoencoder(inputs,block,residual,attention,flat_latent_dim):
    input_shape=input_shape_dict[block]
    x=get_encoder(inputs,input_shape,residual,attention,flat_latent_dim=flat_latent_dim)
    dec=make_decoder(x.shape[1:],residual,attention,flat_latent_dim)
    #x = enc(inputs)
    x=dec(x)
    return x

def aegen(block,base_flat_noise_dim=0,residual=True,attention=True,output_blocks=[],art_styles=[],norm="instance"):
    '''makes autoencoder based generator
    '''
    input_shape=input_shape_dict[block]
    inputs = tk.Input(shape=input_shape)
    flat_latent_dim=base_flat_noise_dim+len(art_styles)
    print(inputs.shape)
    x=get_encoder(inputs,input_shape,residual,attention,base_flat_noise_dim=base_flat_noise_dim,norm=norm)
    if len(art_styles)>0:
        class_inputs=tk.Input(shape=(len(art_styles)))
        x=tf.concat([x,class_inputs],axis=-1)
    dec=make_decoder(x.shape[1:],residual,attention,flat_latent_dim,norm=norm)
    x=dec(x)
    x=tk.applications.vgg19.preprocess_input(x)
    if output_blocks==[]:
        output_blocks.append(block)
    vgg=vgg_layers(output_blocks)
    x=vgg(x)
    if len(art_styles)>0:
        inputs=[inputs,class_inputs]
    return Model(inputs, outputs=x,name='aegen')

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
    return Model(inputs=inputs, outputs=x,name='extracted_generator')

if __name__=='__main__':
    for block in input_shape_dict.keys():
        output_blocks=[]
        model=aegen(block,output_blocks=output_blocks,base_flat_noise_dim=4,art_styles=['baroque','impressionism'],norm="batch")
        print(model.input_shape)
        print(block, model.output_shape)
        gen=extract_generator(model,block,output_blocks)
        print(gen.input_shape)