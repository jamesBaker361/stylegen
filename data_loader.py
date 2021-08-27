from string_globals import *
import os
import tensorflow as tf
import random
import numpy as np

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

class ImageBatchData:
    '''
    '''
    def __init__(self):
        return

def get_all_img_paths(block):
    ret=[]
    styles=[s for s in os.listdir('{}/{}'.format(npz_root,block)) if s[0]!='.']
    print(styles)
    for style in styles:
        imgs=os.listdir('{}/{}/{}'.format(npz_root,block,style))
        for i in imgs:
            ret.append('{}/{}/{}/{}'.format(npz_root,block,style,i))
    return ret

def list_to_batches(flat_list,batch_size=16):
    ret=[]
    for x in range(0,1+len(flat_list)//batch_size):
        ret.append(flat_list[x*batch_size: batch_size*(x+1)])
    return ret

def get_dataset(block,batch_size,limit=5000):
    flat_list=get_all_img_paths(block)[:limit]
    random.shuffle(flat_list)
    list_of_batches=list_to_batches(flat_list,batch_size)
    return BatchData(list_of_batches)

def imposter():
    paths=get_all_img_paths(block1_conv1)
    print('total images: {}'.format(len(paths)))
    bad=[]
    for img in paths:
        try:
            features=np.load(img)['features']
        except:
            print('attribute error for {}'.format(img))
            bad.append(img)
            continue
        if features.shape != (256,256,64):
            print(img,features.shape)
            bad.append(img)
    print(len(bad))
    for img in bad:
        splits=img.split('/')
        with open('{}/{}/{}/badboys.sh'.format(npz_root,block1_conv1,splits[2]),'a+') as file:
            file.write('find . -name  \'{}\' -delete\n'.format(splits[3]))
            file.write('rm {}\n'.format(splits[3]))


if __name__ =='__main__':
    imposter()