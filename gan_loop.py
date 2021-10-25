print('hello world myy name is '.format(__name__))

import os
from tensorflow.python.keras.models import Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import pdb
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(gpu)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
import argparse
import cv2
from string_globals import *
from other_globals import *


from generator import vqgan,noise_dim_dcgan,noise_dim_vqgan,dcgen

noise_dim=noise_dim_dcgan
from discriminator import conv_discrim

from data_loader import get_dataset
from timeit import default_timer as timer

EPOCHS=50
BATCH_SIZE=16
LIMIT=5000
PRE_EPOCHS=0
NAME='v1'
USE_GPU=False

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_vector=tf.ones_like(real_output)+tf.random.normal(shape=real_output.shape,mean=0,stddev=.05)
    fake_vector=tf.zeros_like(fake_output)+tf.random.normal(shape=fake_output.shape,mean=.15,stddev=.05)
    real_loss = cross_entropy(real_vector, real_output)
    fake_loss = cross_entropy(fake_vector, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-2)
discriminator_optimizer = tf.keras.optimizers.SGD()

gen=vqgan()
disc=conv_discrim()

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
#@tf.function
def train_step(images,gen_training=True,disc_training=True):
    batch_size=images.shape[0]
    noise = tf.random.normal([batch_size, *noise_dim])
    #noise = tf.random.normal([batch_size, *noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen(noise, training=gen_training)

        real_output = disc(images, training=disc_training)
        fake_output = disc(generated_images, training=disc_training)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    if gen_training is True:
        gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    if disc_training is True:
        gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))

    return disc_loss,gen_loss

def train(dataset,epochs=EPOCHS,picture=True,pre_train_epochs=PRE_EPOCHS,name=NAME,save_gen=True,save_disc=True):
    check_dir_gen='./{}/{}/{}'.format(checkpoint_dir,name,'gen')
    check_dir_disc='./{}/{}/{}'.format(checkpoint_dir,name,'disc')
    picture_dir='./{}/{}'.format(gen_img_dir,name)
    for d in [check_dir_gen,check_dir_disc,picture_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    print('pretraining')
    for epoch in range(pre_train_epochs):
        dataset.reset()
        for i,images in enumerate(dataset):
            disc_loss,_=train_step(images,gen_training=False)
            if i % 10 == 0:
                print('\tbatch {} disc loss {}'.format(i,disc_loss))
        print('epoch: {} ended with disc_loss {}'.format(epoch,disc_loss))
        if disc_loss<=0.001:
            print('discriminator converged too quickly')
            break
        if save_disc is True:
            save_dir=check_dir_disc+'/pretrain_epoch_{}/'.format(epoch)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            disc.save_weights(save_dir)
    print('training')
    intermediate_model=Model(inputs=gen.input, outputs=gen.get_layer('img_output').output)
    for epoch in range(epochs):
        dataset.reset()
        gen_training=True
        disc_training=True
        for i,images in enumerate(dataset):
            disc_loss,gen_loss=train_step(images,gen_training,disc_training)
            if i%10==0:
                print('\tbatch {} disc_loss: {} gen loss: {}'.format(i,disc_loss,gen_loss))
            if disc_loss <= 0.001:
                disc_training=False
            else:
                disc_training=True
            if gen_loss<=0.001:
                gen_training=False
            else:
                gen_training=True
        print('epoch: {} ended with disc_loss {} and gen loss {}'.format(epoch,disc_loss,gen_loss))
        if save_gen is True:
            save_dir=check_dir_gen+'/epoch_{}/'.format(epoch)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gen.save_weights(save_dir)
        if save_disc is True:
            save_dir=check_dir_disc+'/epoch_{}/'.format(epoch)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            disc.save_weights(save_dir)
        if picture is True:
            noise = tf.random.normal([1, *noise_dim])
            gen_img=intermediate_model(noise).numpy()
            cv2.imwrite('./{}/{}/epoch_{}.jpg'.format(gen_img_dir,name,epoch),gen_img[0])

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='get some args')
    batch_size_str='batch_size'
    epochs_str='epochs'
    pretrain_epochs_str='pretrain_epochs'
    limit_str='limit'
    name_str='name'
    gpu_str='gpu'
    parser.add_argument('--{}'.format(epochs_str),help='epochs to train in tandem',type=int)
    parser.add_argument('--{}'.format(limit_str),help='how many images in training set',type=int)
    parser.add_argument('--{}'.format(batch_size_str),help='batch size',type=int)
    parser.add_argument('--{}'.format(pretrain_epochs_str),help='epochs to pre train discriminator',type=int)
    parser.add_argument('--{}'.format(name_str),help='name of this versions', type = str)

    args = parser.parse_args()

    arg_vars=vars(args)
    print(arg_vars)
    if arg_vars[batch_size_str] is not None:
        BATCH_SIZE=arg_vars[batch_size_str]
    if arg_vars[epochs_str] is not None:
        EPOCHS=arg_vars[epochs_str]
    if arg_vars[limit_str] is not None:
        LIMIT=arg_vars[limit_str]
    if arg_vars[pretrain_epochs_str] is not None:
        PRE_EPOCHS=arg_vars[pretrain_epochs_str]
    if arg_vars[name_str] is not None:
        NAME=arg_vars[name_str]
    dataset=get_dataset(block1_conv1,BATCH_SIZE,LIMIT)
    print('main loop')
    start=timer()
    train(dataset,EPOCHS,pre_train_epochs=PRE_EPOCHS,name=NAME)
    end=timer()
    print('gpu = {} time elapsed {}'.format(USE_GPU,end-start))