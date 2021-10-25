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

noise_shape_dict={ 
    no_block: (32,32,64),
    block1_conv1:(32, 32, 256),
    block2_conv1:(16,16,256),
    block3_conv1: (8,8,256),
    block4_conv1: (4,4,256),
    block5_conv1: (2,2,256)
}

all_genres=[_ for _ in range(-1,18)]

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