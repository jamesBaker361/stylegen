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

# https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py

noise_dim_vqgan = (32, 32, 3)


def _einsum(a, b, c, x, y):
    einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
    return tf.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[:len(x.shape)])
    y_chars = list(string.ascii_uppercase[:len(y.shape)])
    assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


def default_init(scale):
    return tk.initializers.VarianceScaling(
        scale=1e-10 if scale == 0 else scale,
        mode='fan_avg',
        distribution='uniform')


def nin(x, num_units, init_scale=1.0):
    in_dim = int(x.shape[-1])
    w_init = default_init(init_scale)
    b_init = tk.initializers.Zeros()
    W = tf.Variable(w_init(shape=[in_dim, num_units]), name='W')
    b = tf.Variable(b_init(shape=[num_units]), name='b')
    y = contract_inner(x, W) + b
    '''print(y.shape)
    print(x.shape[:-1]+[num_units])
    x_shape=x.shape[:-1] + [num_units]
    print(y.shape,x_shape)
    assert y.shape == x_shape'''
    return y


class NinLayer(layers.Layer):
    def __init__(self, num_units, init_scale=1.0, **kwargs):
        super(NinLayer, self).__init__(kwargs)
        self.num_units = num_units
        self.w_init = default_init(init_scale)
        self.b_init = tk.initializers.Zeros()

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.W = tf.Variable(
            self.w_init(
                shape=[
                    in_dim,
                    self.num_units]),
            name='W')
        self.b = tf.Variable(self.b_init(shape=[self.num_units]), name='b')

    def call(self, x):
        return contract_inner(x, self.W) + self.b


def attn_block(x):
    B, H, W, C = x.shape
    h = GroupNormalization()(x)
    q = NinLayer(C)(h)
    k = NinLayer(C)(h)
    v = NinLayer(C)(h)

    w = tf.einsum('...hwc,...HWc->...hwHW', q, k) * (int(C) ** (-0.5))
    w = layers.Reshape((-1, H, W, H * W))(w)
    w = tf.nn.softmax(w, -1)
    w = layers.Reshape((H, W, H, W))(w)

    h = tf.einsum('...hwHW,...HWc->...hwc', w, v)
    h = NinLayer(C, init_scale=0.0)(h)

    return x + h


def vqgan(noise_dim=noise_dim_vqgan,m=3):
    '''
    Parameters
    ----------
    first_channel -- int. how many output channels in first convolutional layers.
    m -- int. how many times to upsample and then downsample.
    noise_dim -- int. shape = (H,W,C). shape of random noise.
    z -- int. how many output channels in latent space image representation between encoder and decoder.
    Returns
    -------
    model -- tk.Model. generator to generate images of shape = (256,256,3)
    '''
    output_shape=input_shape_dict[block1_conv1] #(256,256,64)
    H,W,C=noise_dim
    # encoder
    inputs = tk.Input(shape=noise_dim)
    # too computationally expensive to generate a bunch of tensors 256,256,3
    x=layers.Conv2DTranspose(3,(8,8),(8,8),padding='same')(inputs)
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
    x = attn_block(x)
    x = ResNextBlock(kernel_size=(4, 4))(x)
    x = GroupNormalization()(x)
    x = tk.activations.swish(x)

    # decoder
    x = ResNextBlock(kernel_size=(4, 4))(x)
    x = attn_block(x)
    x = ResNextBlock(kernel_size=(4, 4))(x)
    for _ in range(m):
        channels = x.shape[-1]/2
        x = ResNextBlock(kernel_size=(4, 4))(x)
        x = layers.Conv2DTranspose(channels, (1, 1), (2, 2))(x)
        x=layers.BatchNormalization()(x)
        x=layers.Dropout(.2)(x)
        x=layers.LeakyReLU()(x)
    x = GroupNormalization(groups=x.shape[-1] // 4)(x)
    x = tk.activations.swish(x)
    x = layers.Conv2D(3, (1, 1), (1, 1))(x)
    x=layers.Activation('sigmoid')(x)
    x=Rescaling(255,name='img_output')(x)
    x=tk.applications.vgg19.preprocess_input(x)
    vgg=vgg_layers([block1_conv1])
    x=vgg(x)
    return Model(inputs=inputs, outputs=[x])


def bs_gen():
    inputs = tk.Input(shape=noise_dim_vqgan)
    x=layers.Conv2DTranspose(3,(8,8),(8,8),padding='same')(inputs)
    return Model(inputs=inputs, outputs=[x])

if __name__=='__main__':
    bs_gen().summary()