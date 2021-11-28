import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from string_globals import *
from other_globals import *
import tensorflow as tf
import random
import numpy as np
import pandas as pd
from data_processing import vgg_layers


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
    '''print(names)
    for style in styles:
        imgs=os.listdir('{}/{}/{}'.format(npz_root,block,style))
        for i in imgs:
            ret.append('{}/{}/{}/{}'.format(npz_root,block,style,i))'''
    return ret

def data_gen(list_of_batches):
    def _data_gen():
        for batch_paths in list_of_batches:
            batch=tf.stack([np.load(path)['features'] for path in batch_paths])
            yield batch
    return _data_gen

def data_gen_2(flat_list):
    def _data_gen_2():
        for path in flat_list:
            features=np.load(path)['features']
            if len(features.shape)>3:
                features=features[0]
            yield features
    return _data_gen_2

def data_gen_slow(blocks,flat_list):
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

def get_dataset_gen(block,batch_size,limit=5000,styles=[], genres=all_genres): #makes batched dataset from generator
    '''makes batched dataset from generator

    Arguments
    --------
    block -- str. one of [no_block, block1_conv1,,,block5_conv1]
    batch_size -- int. batch size
    limit -- int. how many images
    styles -- [str]. something like ['expressionism', 'realism',,,,] 
    genres -- [int]. [-1,,,17] numbers can range from 1 to 20
    '''
    if len(styles)==0:
        styles=[s for s in os.listdir('{}/{}'.format(npz_root,block)) if s[0]!='.']
    flat_list=get_all_img_paths(block,styles,genres)
    random.shuffle(flat_list)
    flat_list=flat_list[:limit]
    gen=data_gen_2(flat_list)
    output_sig_shape=input_shape_dict[block]
    return tf.data.Dataset.from_generator(gen,output_signature=(tf.TensorSpec(shape=output_sig_shape))).batch(batch_size)

def get_dataset_gen_slow(blocks,batch_size,limit=5000,styles=all_styles,genres=all_genres_art):
    flat_list=get_all_img_paths(no_block,styles,genres) #no_block
    random.shuffle(flat_list)
    flat_list=flat_list[:limit]
    flat_list=flat_list[:batch_size*(len(flat_list)//batch_size)]
    print('images in dataset = {}'.format(len(flat_list)))
    gen=data_gen_slow(blocks,flat_list)
    output_sig_shapes=tuple([tf.TensorSpec(shape=input_shape_dict[block]) for block in blocks])
    print(output_sig_shapes)
    return tf.data.Dataset.from_generator(gen,output_signature=output_sig_shapes).batch(batch_size)

def get_real_imgs_fid(block,styles,limit=1000): #gets real images to use as real dataset to compare to generated images for FID
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
    data=get_dataset_gen_slow([no_block],2,limit=200000,styles=['cubism'],genres=all_genres_art)