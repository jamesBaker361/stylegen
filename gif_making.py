import imageio
from string_globals import *
import os
import sys

def make_gif(name):
    print(name)
    target_dir='{}/{}'.format(gen_img_dir,name)
    filenames=[os.path.join(target_dir,f) for f in os.listdir(target_dir) if f.endswith('.jpg')]
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('{}/movie.gif'.format(target_dir), images,duration=len(images)*.02)

if __name__ =='__main__': #pass name ae, dcgen, etc, whatever the name of the folder is as arg
    cl_args=sys.argv[1:]
    prefix=cl_args[0]
    for name in [prefix+'{}'.format(i) for i in range(6)]:
        make_gif(name)