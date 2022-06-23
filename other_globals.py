from tensorflow.keras.applications import VGG19
from string_globals import *
IMG_H=256
IMG_W=256
IMG_C=3
image_dim=(IMG_H,IMG_W,IMG_C)
vgg = VGG19(include_top=False, weights='imagenet',input_shape=image_dim)

input_shape_dict={l.name:l.input_shape[1:] for l in vgg.layers}
input_shape_dict[no_block]=image_dim
input_shape_dict[no_block_raw]=image_dim

all_genres=[_ for _ in range(-1,18)]
all_genres_art=[_ for _ in range(0,18)] #genre -1 is photographs of weird shit
base_flat_noise_dim=256 #the noise dim without any conditionals

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