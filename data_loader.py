import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from string_globals import *
from other_globals import *
import tensorflow as tf
import random
import numpy as np
import pandas as pd
from data_processing import vgg_layers
from sklearn.preprocessing import OneHotEncoder


def get_all_img_paths(block,styles,genres): #gets all image path of images of style/genre (default)
    '''
    Parameters
    ----------
    block -- str.
    styles --[str].
    genres -- [int].
    '''
    ret=[]
    df=pd.read_csv('art_labels.csv')
    int_styles=[]
    for s in styles:
        if s in styles_to_int_dict:
            int_styles.append(styles_to_int_dict[s])
    names=[n+'.npz' for n in df[(df['genre_class_label'].isin(genres)) & (df['style_class_label'].isin(int_styles))]['image_file_name'] ]
    for n in names:
        new_file='{}/{}/{}'.format(npz_root,block,n)
        if os.path.exists(new_file):
            ret.append(new_file)
    return ret

def data_gen_slow(blocks,flat_list):
    '''This function takes in a list of paths to numpy arrays that contain the features of the images. 
    It then takes those features and runs them through the vgg19 model and returns the output of the
    model.
    
    Parameters
    ----------
    blocks
        A list of the layers to be used in the model.
    flat_list
        list of paths to numpy files
    
    Returns
    -------
        A function that returns a generator that yields a tuple of tensors.
    
    '''
    def _data_gen_slow():
        vgg=vgg_layers(blocks)
        for path in flat_list:
            features=np.load(path)['features']
            ''' these should all be no_block but technically they COULD be any block
            like it would be redundant to run a the output of  a feature layer through vgg but we could
            '''
            features=tf.keras.applications.vgg19.preprocess_input(features)
            yield tuple([f for f in vgg(features)])
    return _data_gen_slow

def get_dataset_gen_slow(blocks,batch_size,limit=5000,styles=all_styles,genres=all_genres_art):
    '''makes batched dataset from generator

    Arguments
    --------
    block -- str. one of [no_block, block1_conv1,,,block5_conv1]
    batch_size -- int. batch size
    limit -- int. how many images
    styles -- [str]. something like ['expressionism', 'realism',,,,] 
    genres -- [int]. [-1,,,17] numbers can range from 1 to 20

    Returns
    -------

    dataset --tf.Data.Dataset.
    '''
    flat_list=get_all_img_paths(no_block,styles,genres) #no_block
    random.shuffle(flat_list)
    flat_list=flat_list[:limit]
    flat_list=flat_list[:batch_size*(len(flat_list)//batch_size)]
    print('images in dataset = {}'.format(len(flat_list)))
    gen=data_gen_slow(blocks,flat_list)
    output_sig_shapes=tuple([tf.TensorSpec(shape=input_shape_dict[block]) for block in blocks])
    print(output_sig_shapes)
    return tf.data.Dataset.from_generator(gen,output_signature=output_sig_shapes).batch(batch_size,drop_remainder=True)


def data_gen_slow_labels(blocks,flat_list,one_hot):
    """also returns the class labels, encoded by a one-hot encoder

    Parameters:
    ----------
    blocks -- [str]. the blocks of vgg
    flat_list -- [str]. list of paths of all the npz files
    one_hot -- OneHotEncoder. already fit to the artistic style labels
    """
    def _data_gen_slow_labels():
        vgg=vgg_layers(blocks)
        for path in flat_list:
            npz_object=np.load(path)
            features=npz_object['features']
            ''' these should all be no_block but technically they COULD be any block
            like it would be redundant to run a the output of  a feature layer through vgg but we could
            '''
            features=tf.keras.applications.vgg19.preprocess_input(features)
            artistic_style_encoding=one_hot.transform([[str(npz_object['style'])]]).toarray()[0]
            if len(blocks)==1:
                yield tuple([f for f in vgg(features)]+[artistic_style_encoding])
            else:
                yield tuple([f[0] for f in vgg(features)]+[artistic_style_encoding])
    return _data_gen_slow_labels


def get_dataset_gen_slow_labels(blocks,batch_size, one_hot,limit=5000,styles=all_styles,genres=all_genres_art):
    '''makes batched dataset from generator that also provides label encoding

    Arguments
    --------
    block -- str. one of [no_block, block1_conv1,,,block5_conv1]
    batch_size -- int. batch size
    limit -- int. how many images
    styles -- [str]. something like ['expressionism', 'realism',,,,] 
    genres -- [int]. [-1,,,17] numbers can range from 1 to 20
    one_hot -- OneHotEncoder. fit to the styles

    Returns
    -------

    dataset -- tf.data.Dataset.
    '''
    flat_list=get_all_img_paths(no_block,styles,genres) #no_block
    random.shuffle(flat_list)
    flat_list=flat_list[:limit]
    flat_list=flat_list[:batch_size*(len(flat_list)//batch_size)]
    print('images in dataset = {}'.format(len(flat_list)))
    gen=data_gen_slow_labels(blocks,flat_list,one_hot)
    output_sig_shapes=tuple([tf.TensorSpec(shape=input_shape_dict[block]) for block in blocks]+[tf.TensorSpec(shape=(len(styles)))])
    return tf.data.Dataset.from_generator(gen,output_signature=output_sig_shapes).batch(batch_size,drop_remainder=True)

def get_real_imgs_fid(block,styles,limit=1000): #gets real images to use as real dataset to compare to generated images for FID
    '''It takes a block and a list of styles, and returns a tensor of real images
    
    Parameters
    ----------
    block
        the block of the dataset to use.
    styles
        a list of style names to use for the generated images. If this is empty, then the style will be
    randomly selected from the style folder.
    limit, optional
        the number of images to use for the dataset. If set to 0, all images will be used.
    
    Returns
    -------
        a tensor of shape (1000,64,64,3)
    
    '''
    if len(styles)==0:
        styles=[s for s in os.listdir('{}/{}'.format(npz_root,block)) if s[0]!='.']
    flat_list=get_all_img_paths(no_block,styles)
    random.shuffle(flat_list)
    flat_list=flat_list[:limit]
    all_features=[]
    for path in flat_list:
        features=np.load(path)['features']
        if len(features.shape)>3:
            features=features[0]
        all_features.append(features)
    return tf.stack(all_features)

if __name__=='__main__':
    print('balls')
    styles=["baroque","romanticism","northern-rennaissance"]
    dataset,one_hot=get_dataset_gen_slow_labels([no_block],3,limit=5,styles=styles)
    for d in dataset:
        print(d)