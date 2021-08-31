
import os
from numpy.core.defchararray import endswith
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

from PIL import Image
import numpy as np
import cv2
from string_globals import *
from other_globals import *

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
    '''    if img.ndim > 3:
        img=tf.squeeze(img,axis=0)'''
    return img

def load_img2(path_to_img,max_dim=256):
    image_array=cv2.imread(path_to_img)
    h,w,c=image_array.shape
    scale=max_dim/max(h,w)
    reshaped_img=cv2.resize(image_array,(int(w*scale),int(h*scale)))
    if h<w:
        while h<w:
            h,w,c =reshaped_img.shape
            reshaped_img=cv2.vconcat([reshaped_img,reshaped_img])
    else:
        while w<h:
            h,w,c=reshaped_img.shape
            reshaped_img=cv2.hconcat([reshaped_img,reshaped_img])
    return reshaped_img[0:max_dim,0:max_dim]

def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def get_img_paths(artistic_styles=[]):
    all_imgs=[]
    img_dir='images'
    if artistic_styles==[]:
        artistic_styles=[s for s in os.listdir(img_dir) if s[0]!='.']
    for style in artistic_styles:
        imgs=os.listdir('{}/{}'.format(img_dir,style))
        for i in imgs:
            all_imgs.append('{}/{}/{}'.format(img_dir,style,i))
    return all_imgs

def main():
    artistic_styles=['expressionism']
    count=len(get_img_paths(artistic_styles))
    c=0
    style_layers = ['block1_conv1'] #,'block2_conv1']
    style_extractor = vgg_layers(style_layers)
    img_dir='images'
    for style in artistic_styles:
        imgs=[i for i in os.listdir('{}/{}'.format(img_dir,style)) if i.endswith('jpg')]
        for i in imgs:
            print('{}/{}'.format(c,count))
            c+=1
            img_path='{}/{}/{}'.format(img_dir,style,i)
            try:
                img=load_img2(img_path)
            except AttributeError:
                print(img_path)
                continue
            img_tensor=tf.constant(img)
            img_tensor=tf.reshape(img_tensor,(1, *img_tensor.shape))
            img_tensor=tf.keras.applications.vgg19.preprocess_input(img_tensor)
            style_outputs = style_extractor(img_tensor)
            for layer,output in zip(style_layers,style_outputs):
                out_dir='{}/{}/{}'.format(npz_root,layer,style)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                path='./{}/{}'.format(out_dir,i)
                proper_shape=input_shape_dict[layer]
                if output.shape==proper_shape:
                    np.savez(path,layer=layer,style=style,features=output)
            

if __name__ == '__main__':
    main()