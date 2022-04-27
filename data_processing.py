
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

import numpy as np
import cv2
from string_globals import *
from other_globals import *

def load_img(path_to_img, max_dim=256):
    '''Loads the image from the given path, scales it to a square, and randomly crops a square of the given
    max dimension
    
    Parameters
    ----------
    path_to_img
        the path to the image you want to load.
    max_dim, optional
        The maximum dimension of the image.
    
    '''
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = min(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(1+(shape * scale), tf.int32)
    print(new_shape)

    img = tf.image.resize(img, new_shape,method='lanczos5')
    img=tf.image.random_crop(img,size=(max_dim,max_dim,3))
    return img

def load_img2(path_to_img,max_dim=256):
    '''Loads an image from a file path, resizes it to a max_dim, and then converts it to a numpy array
    
    Parameters
    ----------
    path_to_img
        the path to the image you want to load.
    max_dim, optional
        the max image size in pixels
    
    Returns
    -------
        an image array of the loaded image.
    
    '''
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

    outputs = [vgg.get_layer(name).output if name != no_block else vgg.layers[0].output for name in layer_names ]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def get_img_paths(artistic_styles=[]):
    '''Given a list of artistic styles, return a list of all the image paths in the img_dir directory
    
    Parameters
    ----------
    artistic_styles
        A list of the artistic style you want to use. If this is empty, it will use all the styles in the
    folder.
    
    Returns
    -------
        A list of strings, where each string is the path to an image.
    
    '''
    all_imgs=[]
    if artistic_styles==[]:
        artistic_styles=[s for s in os.listdir(img_dir) if s[0]!='.']
    for style in artistic_styles:
        imgs=os.listdir('{}/{}'.format(img_dir,style))
        for i in imgs:
            all_imgs.append('{}/{}/{}'.format(img_dir,style,i))
    return all_imgs

def main(blocks):
    '''It takes a list of layers, and for each layer, it takes all the images in the img_dir, and saves the
    output of that layer for each image in a .npz file
    
    Parameters
    ----------
    blocks
        a list of strings, each of which is a layer in the VGG19 network.
    
    '''
    artistic_styles= all_styles #['baroque','early-renaissance','high-renaissance','mannerism-late-renaissance','northern-renaissance','ukiyo-e','rococo','realism','contemporary-realism','color-field-painting']
    count=len(get_img_paths(artistic_styles))
    c=0
    style_layers = blocks #,'block2_conv1']
    use_styles=True
    if no_block in set(blocks):
        use_styles=False
    style_extractor={}
    print(use_styles)
    if use_styles==True:
        style_extractor = vgg_layers(style_layers)
    for style in artistic_styles:
        imgs=[i for i in os.listdir('{}/{}'.format(img_dir,style)) if i.endswith('jpg')]
        for i in imgs:
            print('{}/{}'.format(c,count))
            c+=1
            img_path='{}/{}/{}'.format(img_dir,style,i)
            try:
                img=load_img(img_path)
            except (AttributeError, ValueError):
                print(img_path)
                continue
            img_tensor=tf.constant(img)
            img_tensor=255*tf.reshape(img_tensor,(1, *img_tensor.shape))
            if use_styles==True:
                img_tensor=tf.keras.applications.vgg19.preprocess_input(img_tensor)
                style_outputs = style_extractor(img_tensor)
            else:
                style_outputs=[img_tensor]
            for layer,output in zip(style_layers,style_outputs):
                out_dir='{}/{}/{}'.format(npz_root,layer,style)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                path='./{}/{}'.format(out_dir,i)
                if os.path.exists(path) == False:
                    proper_shape=input_shape_dict[layer]
                    print(layer,output.shape,proper_shape)
                    if output.shape==proper_shape or (output.shape[1:]==proper_shape):
                        np.savez(path,layer=layer,style=style,features=output)
                        print('saved ',path)
                    else:
                        print('output.shape!=proper_shape',path)
                else:
                    print('exists ',path)
            

if __name__ == '__main__':
    blocks=[]
    for arg in set(sys.argv):
        if arg in input_shape_dict:
            blocks.append(arg)
    main(blocks)