
import os
npz_root='../../../../../scratch/jlb638/imgs_npz' #where the npz matrices are; CHANGE THIS FOR UR OWN USES
img_dir='../../../../../scratch/jlb638/images' #where the actual images themselves are; CHANGE THIS
checkpoint_dir='../../../../../scratch/jlb638/checkpoints' #where saved models are stored CHANGE THIS FOR YOUR OWN PURPOSES
all_styles=[s for s in os.listdir('{}'.format(img_dir)) if s[0]!='.']
open_img_dir='../../../../../scratch/jlb638/open-imgs' #this is another image dataset
no_block='no_block' #these are just normal images
block1_conv1 ='block1_conv1'
block1_conv2 ='block1_conv2'
block1_pool ='block1_pool'
block2_conv1 ='block2_conv1'
block2_conv2 ='block2_conv2'
block2_pool ='block2_pool'
block3_conv1 ='block3_conv1'
block3_conv2 ='block3_conv2'
block3_conv3 ='block3_conv3'
block3_conv4 ='block3_conv4'
block3_pool ='block3_pool'
block4_conv1 ='block4_conv1'
block4_conv2 ='block4_conv2'
block4_conv3 ='block4_conv3'
block4_conv4 ='block4_conv4'
block4_pool ='block4_pool'
block5_conv1 ='block5_conv1'
block5_conv2 ='block5_conv2'
block5_conv3 ='block5_conv3'
block5_conv4 ='block5_conv4'
block5_pool ='block5_pool'
all_blocks=[no_block,block1_conv1,block1_conv2,block1_pool,block2_conv1,block2_conv2,block2_pool,block3_conv1,block3_conv2,block3_conv3,block3_conv4,block3_pool,block4_conv1,block4_conv2,block4_conv3,block4_conv4,block4_pool,block5_conv1,block5_conv2,block5_conv3,block5_conv4,block5_pool]
gen_img_dir='./gen_imgs' #generated images
graph_dir='./graphs' #graphs showing performance
stylized_img_dir='./stylized_imgs' #the images after the have been sttylized
#raw_image_dir='/scratch/jlb638/images/