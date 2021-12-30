from string_globals import *
image_dim=(256,256,3)

input_shape_dict={ #shape of input to discriinator
    no_block: image_dim,
    block1_conv1:(256, 256, 64),
    block2_conv1:(128,128,128),
    block3_conv1: (64,64,256),
    block4_conv1: (32,32,512),
    block5_conv1: (16,16,512)
}

input_shape_dict_big={
    no_block: image_dim ,
block1_conv1 : (256, 256, 64) ,
block1_conv2 : (256, 256, 64) ,
block1_pool : (128, 128, 64) ,
block2_conv1 : (128, 128, 128) ,
block2_conv2 : (128, 128, 128) ,
block2_pool : (64, 64, 128) ,
block3_conv1 : (64, 64, 256) ,
block3_conv2 : (64, 64, 256) ,
block3_conv3 : (64, 64, 256) ,
block3_conv4 : (64, 64, 256) ,
block3_pool : (32, 32, 256) ,
block4_conv1 : (32, 32, 512) ,
block4_conv2 : (32, 32, 512) ,
block4_conv3 : (32, 32, 512) ,
block4_conv4 : (32, 32, 512) ,
block4_pool : (16, 16, 512) ,
block5_conv1 : (16, 16, 512) ,
block5_conv2 : (16, 16, 512) ,
block5_conv3 : (16, 16, 512) ,
block5_conv4 : (16, 16, 512) ,
block5_pool : (8, 8, 512)
}
input_shape_dict=input_shape_dict_big
all_genres=[_ for _ in range(-1,18)]
all_genres_art=[_ for _ in range(0,18)] #genre -1 is photographs of weird shit
base_flat_noise_dim=128 #the noise dim without any conditionals

styles_to_int_dict = {'early-renaissance' : 1,
'high-renaissance' : 2,
'mannerism-late-renaissance' : 3,
'northern-renaissance' : 4,
'baroque' : 5,
'rococo' : 6,
'romanticism' : 7,
'impressionism' : 8,
'pointillism' : 9,
'realism' : 10,
'contemporary-realism':10,
'art-nouveau-modern' : 11,
'cubism' : 12,
'expressionism' : 13,
'symbolism' : 14,
'fauvism' : 15,
'abstract-expressionism' : 16,
'color-field-painting' : 17,
'minimalism' : 18,
'na-ve-art-primitivism' : 19,
'ukiyo-e' : 20,
'pop-art' : 21
}