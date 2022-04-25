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

def get_encoder(inputs,input_dim,residual,
    noise_weight=1.0,
    base_flat_noise_dim=0,norm="instance"):
    '''> The encoder takes an input image and returns a vector of size `base_flat_noise_dim`
    
    The encoder is a convolutional neural network that takes an image as input and returns a vector of
    size `base_flat_noise_dim` if base_flat_noise_dim !=0. Else: The encoder is a series of convolutional layers that downsample the
    image until it is 2x2x (Dim), where dim is variable
    
    Parameters
    ----------
    inputs
        the input tensor
    input_dim
        the shape of the input image
    residual
        whether to use residual blocks
    attention
        whether to use attention in the decoder (depr)
    noise_weight
        the amount to scale the noise to add to the encoder output.
    m, optional
        the number of layers in the decoder (depr)
    base_flat_noise_dim, optional
        the dimension of the flat noise vector.
    norm, optional
        "instance", "batch", or "group"
    
    Returns
    -------
        The encoder is returning a tensor of shape (batch_size, base_flat_noise_dim)
    
    '''
        
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
    
    while x.shape[-1]<64:
        channels = x.shape[-1] *2
        x=block(x, channels)
    
    while x.shape[-1]>64:
        channels = x.shape[-1] //2
        x=block(x, channels)
    while x.shape[-2] >2:
        channels = x.shape[-1]
        if residual==True:
            x = ResNextBlock(kernel_size=(4, 4))(x)
        #print(x.shape)
        x = layers.Conv2D(channels, (4, 4), (2, 2), padding='same')(x)
        x=normalization()(x)
        ##x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        if residual==True:
            x = ResNextBlock(kernel_size=(4, 4))(x)
        x=layers.Conv2D(channels,(2,2),padding='same')(x)
        x=normalization()(x)
        ##x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
    if noise_weight!=0:
        noise=noise_weight*tf.random.normal(x.shape[1:])
        x=tf.keras.layers.Add()([x,noise])

    if base_flat_noise_dim>0:
        x=layers.Flatten()(x)
        x=layers.Dense(base_flat_noise_dim)(x)
    return x

def make_decoder(input_dim,residual,attention,flat_latent_dim=0,norm="instance"):
    '''It takes an input, and then applies a series of convolutions and transposed convolutions to it, with
    some residual blocks and attention blocks thrown in
    
    Parameters
    ----------
    input_dim
        the shape of the input image
    residual
        Whether to use residual blocks in the decoder.
    attention
        whether to use attention blocks
    flat_latent_dim, optional
        if you flat_latent_dim !=0, then it must be reshaped to be (2,2,64)
    norm, optional
        "instance", "batch", or "group"
    
    Returns
    -------
        A model with the input and output layers defined.
    
    '''
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
        x = layers.Conv2DTranspose(channels, (4, 4), (2, 2),padding='same')(x)
        x=normalization()(x)
        ##x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
        if residual==True:
            x = ResNextBlock(kernel_size=(4, 4))(x)
        if x.shape[-2]<64 and attention==True:
            x=attn_block(x)
    x = layers.Conv2D(x.shape[-1], (8, 8), (1, 1),padding='same')(x)
    x = GroupNormalization(groups=x.shape[-1] // 4)(x)
    x = tk.activations.swish(x)
    x = layers.Conv2D(8, (8, 8), (1, 1),padding='same')(x)
    x = GroupNormalization(groups=x.shape[-1] // 4)(x)
    x = tk.activations.swish(x)
    x = layers.Conv2D(4, (4, 4), (1, 1),padding='same')(x)
    x = GroupNormalization(groups=x.shape[-1] // 4)(x)
    x = tk.activations.swish(x)
    x = layers.Conv2D(3, (4, 4), (1, 1),padding='same')(x)
    x=layers.Activation('sigmoid')(x)
    x=Rescaling(255,name='img_output')(x)
    return tk.Model(inputs=inputs,outputs=x,name='decoder')

def full_autoencoder(inputs,block,residual,attention,flat_latent_dim,noise_weight):
    '''> The function takes in the inputs, the block, the residual, the attention, and the flat latent
    dimension, and returns the full autoencoder
    
    Parameters
    ----------
    inputs
        the input tensor
    block
        the VGG block this takes as input
    residual
        Whether to use residual connections in the encoder and decoder.
    attention
        whether to use attention in the encoder and decoder
    flat_latent_dim
        the dimension of the latent space.
    noise_weight
        the amount to scale the noise to add to the encoder output.
    
    Returns
    -------
        The output of the decoder.
    
    '''
    input_shape=input_shape_dict[block]
    x=get_encoder(inputs,input_shape,residual,noise_weight=noise_weight,flat_latent_dim=flat_latent_dim)
    dec=make_decoder(x.shape[1:],residual,attention,flat_latent_dim)
    #x = enc(inputs)
    x=dec(x)
    return x

def aegen(block,base_flat_noise_dim=0,residual=True,attention=True,output_blocks=[],art_styles=[],norm="instance",noise_weight=1.0):
    '''`aegen` takes in an image, encodes it, adds some noise, decodes it, and then compares the decoded
    image to the original image
    
    Parameters
    ----------
    block
        the block of the vgg19 model to use as the output.
    base_flat_noise_dim, optional
        The dimension of the latent space that is not used for the art style.
    residual, optional
        Whether to use residual blocks in the encoder and decoder
    attention, optional
        Whether to use attention in the encoder and decoder
    output_blocks
        the layers of the vgg19 model that you want to use as outputs.
    art_styles
        a list of strings, each string is the name of a style.
    norm, optional
        "instance" or "batch"
    noise_weight, optional
        the amount to scale the noise to add to the encoder output.
    
    Returns
    -------
        The model is being returned.
    
    '''
    input_shape=input_shape_dict[block]
    inputs = tk.Input(shape=input_shape)
    flat_latent_dim=base_flat_noise_dim+len(art_styles)
    print(inputs.shape)
    x=get_encoder(inputs,input_shape,residual,
        noise_weight=noise_weight,base_flat_noise_dim=base_flat_noise_dim,norm=norm)
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