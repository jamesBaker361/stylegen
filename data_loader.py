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
    for s in styles:
        ret+=[os.path.join(npz_root,block,s,f) for f  in os.listdir(os.path.join(npz_root,block,s)) if f.endswith("npz")]
    return ret

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
            ''' these should all be no_block_raw but technically they COULD be any block
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
    flat_list=get_all_img_paths(no_block_raw,styles)
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
    
    styles=["079_teana_lanster","197_illyasviel_von_einzbern"]
    one_hot=OneHotEncoder()
    one_hot.fit([[s] for s in styles])
    dataset=get_dataset_gen_slow_labels([no_block,block1_conv1],3,one_hot,limit=10,styles=styles)
    for d in dataset:
        print(d)