import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import layers,Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers  import GaussianNoise

from tensorflow.python.keras.activations import sigmoid

from group_norm import GroupNormalization
from resnext import ResNextBlock

from data_processing import vgg_layers
from rescaling import Rescaling

from other_globals import *
from generator import *

 #the dim of latent space is dim is 1-D

def get_encoder(inputs,input_dim,residual, attention=False,
    noise_weight=1.0,
    base_flat_noise_dim=0,norm="instance"):
        
    normalization=InstanceNormalization
    if norm == "batch":
        normalization=BatchNormalization
    elif norm == "group":
        normalization=GroupNormalization
        
    def block(x,channels):
        x= layers.Conv2D(channels, (4, 4), (1, 1), padding='same')(x)
        x=normalization()(x)
        ##x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        x=layers.Conv2D(channels,(2,2),padding='same')(x)
        x=normalization()(x)
        ##x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        return x
    x = layers.Conv2D(max(8,input_dim[-1]), (8, 8), (1, 1),padding='same')(inputs)
    x=normalization()(x)
    x=layers.LeakyReLU()(x)
    if residual==True:
        x = ResNextBlock(kernel_size=(4, 4))(x)
    
    
    while x.shape[-2] >16:
        if x.shape[-2]<64 and attention==True:
            x=attn_block(x)
        channels = x.shape[-1]
        if channels<64:
            channels=channels*2
        #print(x.shape)
        x = layers.Conv2D(channels, (4, 4), (2, 2), padding='same')(x)
        x=normalization()(x)
        ##x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        if residual==True:
            x = ResNextBlock(kernel_size=(4, 4))(x)
    while x.shape[-2]>2:
        if x.shape[-2]<64 and attention==True:
            x=attn_block(x)
        if channels<64:
            channels=channels*2
        x = layers.Conv2D(channels, (3, 3), (1, 1))(x)
        x=normalization()(x)
        ##x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        if residual==True:
            x = ResNextBlock(kernel_size=(4, 4))(x)
    while x.shape[-1]>64:
        channels = x.shape[-1]//2
        x=layer.Conv2D(channels,(1,1),(1,1))(x)
        x=normalization()(x)
        x=layers.LeakyReLU()(x)


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
    x = layers.Conv2D(x.shape[-1], (8, 8), (1, 1),padding='same')(x)
    x=normalization()(x)
    ##x=layers.Dropout(.2)(x)
    x=layers.LeakyReLU()(x)
    if attention==True:
        x=attn_block(x)
    while x.shape[-2]<256:
        channels = max(x.shape[-1]//2,32)
        if residual==True:
            x = ResNextBlock(kernel_size=(4, 4))(x)
        x = layers.Conv2DTranspose(channels, (3, 3), (2, 2),padding='same')(x)
        x=normalization()(x)
        ##x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        if residual==True:
            x = ResNextBlock(kernel_size=(4, 4))(x)
        if x.shape[-2]<64 and attention==True:
            x=attn_block(x)
    x = layers.Conv2D(16, (5, 5), (1, 1),padding='same')(x)
    x = GroupNormalization(groups=x.shape[-1] // 4)(x)
    x = tk.activations.swish(x)
    x = layers.Conv2D(8, (3, 3), (1, 1),padding='same')(x)
    x = GroupNormalization(groups=x.shape[-1] // 4)(x)
    x = tk.activations.swish(x)
    x = layers.Conv2D(4, (1, 1), (1, 1),padding='same')(x)
    x = GroupNormalization(groups=x.shape[-1] // 4)(x)
    x = tk.activations.swish(x)
    x = layers.Conv2D(3, (1, 1), (1, 1),padding='same')(x)
    x=layers.Activation('sigmoid')(x)
    x=Rescaling(255,name='img_output')(x)
    return tk.Model(inputs=inputs,outputs=x,name='decoder')

def aegen(block,base_flat_noise_dim=0,residual=True,attention=True,output_blocks=[],art_styles=[],norm="instance",noise_weight=1.0,batch_size=1):
    input_shape=input_shape_dict[block]
    inputs = tk.Input(shape=input_shape)
    flat_latent_dim=base_flat_noise_dim+len(art_styles)
    print(inputs.shape)
    x=get_encoder(inputs,input_shape,residual,
        noise_weight=noise_weight,base_flat_noise_dim=base_flat_noise_dim,norm=norm)
    if noise_weight!=0:
        x=GaussianNoise(noise_weight)(x)
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
    '''It takes the decoder part of the autoencoder, and adds the VGG19 layers to it
    
    Parameters
    ----------
    model
        the model to extract the generator from
    block
        the block number of the VGG19 model to extract features from.
    output_blocks
        a list of the names of the layers you want to extract from the VGG19 model.
    
    Returns
    -------
        A model that takes in the input shape of the decoder and outputs the output of the vgg model.
    
    '''
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