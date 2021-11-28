import os
import random
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import concat
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import numpy as np
import PIL.Image
import time
import functools
import tensorflow_hub as hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
from string_globals import *

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)

content_path='/scratch/jlb638/images/baroque/999_the-man-with-the-slouch-hat.jpg!Blog.jpg'
style_path='/home/jlb638/Desktop/stylegen/gen_imgs/ae_5000_2/epoch_18_iii.jpg'

def style_transfer(content_path,style_path,output_path,save=True):
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0][0]
    if save is True:
        tf.keras.utils.save_img(output_path,stylized_image)
    return stylized_image


if __name__=='__main__':
    style='baroque'
    texture='ae_10000_2'
    for style in all_styles:
        for texture in ['ae_5000_{}'.format(j) for j in range(6)]+['ae_10000_{}'.format(j) for j in range(6)]:
            content_src_dir='/scratch/jlb638/images/{}/'.format(style) #the artistic images
            style_src_dir='/home/jlb638/Desktop/stylegen/gen_imgs/{}/'.format(texture) #the generated textures
            output_dir= '{}/{}/{}'.format(stylized_img_dir,style,texture)  #stylized_img_dir+'/baroque/ae_10000_2'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            style_q=12
            content_paths=[content_src_dir+c for c in  random.sample(os.listdir(content_src_dir),10)]
            style_paths=[style_src_dir+s for s in  random.sample(os.listdir(style_src_dir),style_q)]
            for i,c_path in enumerate(content_paths):
                try:
                    images=[]
                    for s_path in style_paths:
                        images.append(style_transfer(c_path,s_path,'_.png',False))
                    h=4
                    rows=[]
                    for x in range(0,style_q,h):
                        rows.append(tf.concat(images[x:x+h],1))
                    concat_imgs=tf.concat(rows,0)
                    output_path='{}/{}.png'.format(output_dir,i)
                    tf.keras.utils.save_img(output_path,concat_imgs)
                except ValueError:
                    print('value error for ',style_path,content_path)