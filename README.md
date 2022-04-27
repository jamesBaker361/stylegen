# ARTEMIS

## Setup
Configure where you're storing your images/checkpoints/VGG extractions in `string_globals.py`
Make sure your image dir has subdirs for each style of image. For example:

img_dir \
 |-- high-renaissance \
 |-- mannerism-late-renaissance \
 |--northern-renaissance \
 |--baroque \
 |-- etc. \

In all of my experiments, I used the wikiart dataset, available here: https://drive.google.com/file/d/182-pFiKvXPB25DbTfAYjJ6gDE-ZCRXz0/view

Run `python data_processing.py ARG` where ARG is any string in `no_block,block1_conv1,block1_conv2,block1_pool,block2_conv1,block2_conv2,block2_pool,block3_conv1,block3_conv2,block3_conv3,block3_conv4,block3_pool,block4_conv1,block4_conv2,block4_conv3,block4_conv4,block4_pool,block5_conv1,block5_conv2,block5_conv3,block5_conv4,block5_pool` this will create the npz files that the GAN actually trains upon.

## Running
Use `gan_loop_mirror_copy_multi.py` to run experiments
run `python gan_loop_mirror_copy_multi.py --help` will print out this:
```
optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       epochs to train generator/discriminators in tandem (int)
  --limit LIMIT         how many images in training set (int)
  --batch_size_replica BATCH_SIZE_REPLICA
                        batch size (int)
  --pretrain_epochs PRETRAIN_EPOCHS
                        epochs to pre train discriminator (int)
  --name NAME           name of this versions (str)
  --block BLOCK         block of vgg we are training autoencoder with (str)
  --auto AUTO           whetther to use an autoencoder GAN (bool)
  --fid FID             whether to calculate fid score after each epoch (bool)
  --human HUMAN         only using the human art (bool)
  --baroque BAROQUE     only using baroque and romantic styles (bool)
  --renn RENN           only using rennaissance styles (bool)
  --flat FLAT           flat latent space for noise (bool)
  --half HALF           whether to only use half the whole dataset for speed purposes (bool)
  --no_load NO_LOAD     whether to load past pretrained versions of models (bool)
  --no_attn NO_ATTN     whether to not use attentional blocks or not (bool)
  --no_res NO_RES       whether to not use resnext blocks (bool, deprecated)
  --ae_epochs AE_EPOCHS
                        how many epochs to train autoencoder for (int)
  --no_diversity NO_DIVERSITY
                        whether to train the generator to maximize diversity of samples (bool)
  --beta BETA           beta coefficient on diversity term (float)
  --conditional CONDITIONAL
                        whether to make it a conditional GAN or not (bool)
  --gamma GAMMA         gamma coefficient on classification loss (float)
  --output_blocks OUTPUT_BLOCKS [OUTPUT_BLOCKS ...]
  --norm NORM           instance batch or group (str)
  --ukiyo UKIYO         whether to only use ukiyo art (bool)
  --encoder_noise ENCODER_NOISE
                        what to multiply the encoder noise by (float)
```

Maany of these optional arguments (`renn,human,baroque,ukiyo`) are only applicable to the wikiArt dataset. Ignore them if using a different dataset.

## Details
This code is optimized for, but does not necessarily need to run on multiple GPUs
I used Python 3.9.7, tensorflow 2.7.0 with the following packages:
```
_libgcc_mutex=0.1
_openmp_mutex=4.5
absl-py=1.0.0
astunparse=1.6.3
backcall=0.2.0
backports=1.0
backports.functools_lru_cache=1.6.4
ca-certificates=2021.10.8
cachetools=4.2.4
certifi=2021.10.8
charset-normalizer=2.0.9
cycler=0.11.0
debugpy=1.5.1
decorator=5.1.0
entrypoints=0.3
flatbuffers=2.0
fonttools=4.28.3
gast=0.4.0
google-auth=2.3.3
google-auth-oauthlib=0.4.6
google-pasta=0.2.0
grpcio=1.42.0
h5py=3.6.0
idna=3.3
imageio=2.13.3
importlib-metadata=4.8.2
ipykernel=6.6.0
ipython=7.30.1
jedi=0.18.1
joblib=1.1.0
jupyter_client=7.1.0
jupyter_core=4.9.1
keras=2.7.0
keras-preprocessing=1.1.2
kiwisolver=1.3.2
ld_impl_linux-64=2.35.1
libclang=12.0.0
libffi=3.3
libgcc-ng=9.3.0
libgomp=9.3.0
libsodium=1.0.18
libstdcxx-ng=9.3.0
markdown=3.3.6
matplotlib=3.5.1
matplotlib-inline=0.1.3
ncurses=6.3
nest-asyncio=1.5.4
numpy=1.21.4
oauthlib=3.1.1
opencv-python=4.5.4.60
openssl=1.1.1l
opt-einsum=3.3.0
packaging=21.3
pandas=1.3.4
parso=0.8.3
pexpect=4.8.0
pickleshare=0.7.5
pillow=8.4.0
pip=21.2.4
prompt-toolkit=3.0.24
protobuf=3.19.1
ptyprocess=0.7.0
pyasn1=0.4.8
pyasn1-modules=0.2.8
pygments=2.10.0
pyparsing=3.0.6
python=3.9.7
python-dateutil=2.8.2
python_abi=3.9
pytz=2021.3
pyzmq=19.0.2
readline=8.1
requests=2.26.0
requests-oauthlib=1.3.0
rsa=4.8
scikit-learn=1.0.1
scipy=1.7.3
setuptools=58.0.4
six=1.16.0
sqlite=3.36.0
tensorboard=2.7.0
tensorboard-data-server=0.6.1
tensorboard-plugin-wit=1.8.0
tensorflow=2.7.0
tensorflow-addons=0.15.0
tensorflow-estimator=2.7.0
tensorflow-hub=0.12.0
tensorflow-io-gcs-filesystem=0.22.0
termcolor=1.1.0
threadpoolctl=3.0.0
tk=8.6.11
tornado=6.1
traitlets=5.1.1
typeguard=2.13.3
typing-extensions=4.0.1
tzdata=2021e
urllib3=1.26.7
wcwidth=0.2.5
werkzeug=2.0.2
wheel=0.37.0
wrapt=1.13.3
xz=5.2.5
zeromq=4.3.4
zipp=3.6.0
zlib=1.2.11
```

