from string_globals import *
from other_globals import *
import os
import tensorflow as tf
import random
import numpy as np
import pandas as pd

class BatchData:
    '''
    for loading intermediate style represenetations'''
    def __init__(self,list_of_batches):
        self.i=0
        self.list_of_batches= [b for b in list_of_batches if len(b)>0]

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >=len(self.list_of_batches):
            raise StopIteration
        batch_paths=self.list_of_batches[self.i]
        batch=tf.stack([np.load(path)['features'] for path in batch_paths])
        self.i+=1
        return batch

    def __len__(self):
        return len(self.list_of_batches)

    def reset(self):
        self.i=0

def get_all_img_paths(block,styles,genres): #gets all image path of images of particular style (default)
    ret=[]
    df=pd.read_csv('art_labels.csv')
    int_styles=[styles_to_int_dict[s] for s in styles]
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

def get_all_img_paths_classlabels(block,column,labels): #gets all image paths of images of particular artists
    '''gets all image paths of images of particular artists

    Arguments:
    ---------
    block -- str. one of [no_block,block1_conv1,,,,block5_conv1]
    column -- str. one of [style_class_label,genre_class_label,artist_class_label]
    '''
    df=pd.read_csv('art_labels.csv')
    return

def get_all_img_paths_genre(block,genres): #gets all image paths of images labeled with particular genres
    '''gets all image path for particualr genres

    Arguments:
    ---------
    block -- str. one of [no_block,block1_conv1,,,,block5_conv1]
    genres -- [str]. any of [-1...17]
    '''
    return

def list_to_batches(flat_list,batch_size=16):
    ret=[]
    for x in range(0,1+len(flat_list)//batch_size):
        ret.append(flat_list[x*batch_size: batch_size*(x+1)])
    return ret

def get_dataset(block,batch_size,limit=5000):
    flat_list=get_all_img_paths(block)
    random.shuffle(flat_list)
    flat_list=flat_list[:limit]
    list_of_batches=list_to_batches(flat_list,batch_size)
    return BatchData(list_of_batches)

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
    '''list_of_batches=list_to_batches(flat_list,1)
    gen= data_gen(list_of_batches)'''
    gen=data_gen_2(flat_list)
    output_sig_shape=input_shape_dict[block]
    return tf.data.Dataset.from_generator(gen,output_signature=(tf.TensorSpec(shape=output_sig_shape))).batch(batch_size)

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
    genres=[[_] for _ in all_genres]
    styles=[s for s in os.listdir('{}/{}'.format(npz_root,no_block)) if s[0]!='.']
    for g in genres:
        print(g[0], len(get_all_img_paths(no_block,styles,g)))