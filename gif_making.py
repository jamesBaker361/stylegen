import imageio
from string_globals import *
import os
import sys
import numpy as np
import cv2

def make_gif(name):
    print(name)
    target_dir='{}/{}'.format(gen_img_dir,name)
    filenames=[os.path.join(target_dir,f) for f in os.listdir(target_dir) if f.endswith('.jpg')]
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('{}/movie.gif'.format(target_dir), images,duration=len(images)*.002)

def make_big_gif(name):
    target_dir='{}/{}'.format(gen_img_dir,name)
    filenames=[os.path.join(target_dir,f) for f in os.listdir(target_dir) if f.endswith('.jpg')]
    images = []
    for f in range(0,len(filenames)-9,9):
        collage=[]
        for x in range(3):
            row=[]
            for y in range(3):
                row.append(cv2.imread(filenames[f+x+(y*3)]))
            collage.append(cv2.hconcat(row))
        gen_img_collage=cv2.vconcat(collage)
        #img=np.reshape([cv2.imread(fname) for fname in filenames[f:f+9]],(3,3))
        images.append(gen_img_collage)
    imageio.mimsave('{}/collage_movie.gif'.format(target_dir), images,duration=len(images)*.02)

if __name__ =='__main__': #pass name ae, dcgen, etc, whatever the name of the folder is as arg
    cl_args=sys.argv[1:]
    prefix=cl_args[0]
    for name in [prefix+'{}'.format(i) for i in range(6)]:
        make_big_gif(name)