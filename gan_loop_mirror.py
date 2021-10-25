print('hello world myy name is '.format(__name__))
import os
from tensorflow.python.keras.models import Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
#tf.config.run_functions_eagerly(True)

import argparse
import cv2
from string_globals import *
from other_globals import *


from generator import vqgen,noise_dim_dcgan,noise_dim_vqgen,dcgen,aegen

noise_dim=noise_dim_dcgan
from discriminator import conv_discrim

from data_loader import get_dataset_gen,get_real_imgs_fid
from timeit import default_timer as timer
from keras.applications.inception_v3 import InceptionV3
from fid_metric import calculate_fid

EPOCHS=1 #how mnay epochs to train generator for
AE_EPOCHS=1 #how many epochs to pre train autoencoder for
BATCH_SIZE_PER_REPLICA=1
LIMIT=100 #how many images in total dataset
PRE_EPOCHS=1 #how many epochs to pretrain discriminator on
NAME='testing'
BLOCK=block1_conv1 #which block of vgg we care about
AUTO=False #whether to use autoencoder generator
FID=False #whether to calculate FID score after each epoch



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='get some args')
    batch_size_replica_str='batch_size_replica'
    epochs_str='epochs'
    pretrain_epochs_str='pretrain_epochs'
    limit_str='limit'
    name_str='name'
    gpu_str='gpu'
    block_str='block'
    auto_str='auto'
    fid_str='fid'
    parser.add_argument('--{}'.format(epochs_str),help='epochs to train in tandem',type=int)
    parser.add_argument('--{}'.format(limit_str),help='how many images in training set',type=int)
    parser.add_argument('--{}'.format(batch_size_replica_str),help='batch size',type=int)
    parser.add_argument('--{}'.format(pretrain_epochs_str),help='epochs to pre train discriminator',type=int)
    parser.add_argument('--{}'.format(name_str),help='name of this versions', type = str)
    parser.add_argument('--{}'.format(block_str),help='block of vgg we are trying to imitate',type=str)
    parser.add_argument('--{}'.format(auto_str),help='whetther to use an autoencoder GAN',type=bool)
    parser.add_argument('--{}'.format(fid_str),help='whether to calculate fid score after each epoch',type=str)

    args = parser.parse_args()

    arg_vars=vars(args)
    print(arg_vars)
    if arg_vars[batch_size_replica_str] is not None:
        BATCH_SIZE_PER_REPLICA=arg_vars[batch_size_replica_str]
    if arg_vars[epochs_str] is not None:
        EPOCHS=arg_vars[epochs_str]
    if arg_vars[limit_str] is not None:
        LIMIT=arg_vars[limit_str]
    if arg_vars[pretrain_epochs_str] is not None:
        PRE_EPOCHS=arg_vars[pretrain_epochs_str]
    if arg_vars[name_str] is not None:
        NAME=arg_vars[name_str]
    if arg_vars[block_str] is not None:
        BLOCK= arg_vars[block_str]
    if arg_vars[auto_str] is not None:
        AUTO=arg_vars[auto_str]
    if arg_vars[fid_str] is not None:
        if arg_vars[fid_str] in set(['true','True']):
            FID=True
        elif arg_vars[fid_str] in set(['false','False']):
            FID=False
    
    SHAPE=input_shape_dict[BLOCK]
    art_styles=['realism']
    print(BLOCK,SHAPE)

    physical_devices=tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device,True)

    gpus = tf.config.list_logical_devices('GPU')
    
    strategy = tf.distribute.MirroredStrategy(gpus)

    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    LIMIT=LIMIT-(LIMIT%GLOBAL_BATCH_SIZE)
    print('limit is {}'.format(LIMIT))
    print('limit {} / batch size {} = {}'.format(LIMIT,GLOBAL_BATCH_SIZE,LIMIT/GLOBAL_BATCH_SIZE))

    print('{} * {} = {}'.format(BATCH_SIZE_PER_REPLICA,strategy.num_replicas_in_sync,GLOBAL_BATCH_SIZE))

    with strategy.scope():
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)

        def discriminator_loss(real_output, fake_output):
            real_vector=tf.ones_like(real_output)+tf.random.normal(shape=real_output.shape,mean=0,stddev=.05)
            fake_vector=tf.zeros_like(fake_output)+tf.random.normal(shape=fake_output.shape,mean=.15,stddev=.05)
            real_loss = cross_entropy(real_vector, real_output)
            fake_loss = cross_entropy(fake_vector, fake_output)
            total_loss = real_loss + fake_loss
            return tf.nn.compute_average_loss(total_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        def generator_loss(fake_output):
            loss=cross_entropy(tf.ones_like(fake_output), fake_output)
            return tf.nn.compute_average_loss(loss, global_batch_size=GLOBAL_BATCH_SIZE)

        def autoencoder_loss(images, generated_images):
            loss=[tf.reduce_mean(tf.square(tf.subtract(images, generated_images)))]
            return tf.nn.compute_average_loss(loss,global_batch_size=GLOBAL_BATCH_SIZE)

        generator_optimizer = tf.keras.optimizers.Adam(1e-2)
        discriminator_optimizer = tf.keras.optimizers.SGD()

        if AUTO==True:
            gen=aegen(BLOCK)
        else:
            gen=vqgen(BLOCK)
        disc=conv_discrim(BLOCK)

        noise_dim=gen.input.shape
        if len(noise_dim)>3:
            noise_dim=noise_dim[1:]

    def train_step(images,gen_training,disc_training):
        batch_size=images.shape[0]
        noise = tf.random.normal([batch_size, * noise_dim])
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

    def train_step_ae(images): #training autoencoder to reconstruct things, not generate
        batch_size=images.shape[0]
        with tf.GradientTape() as tape:
            reconstructed_images=gen(images)
            ae_loss=autoencoder_loss(images,reconstructed_images)

        gradients_of_generator = tape.gradient(ae_loss, gen.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
        return ae_loss


    def get_train_step_dist():
        @tf.function
        def train_step_dist(images,gen_training=True,disc_training=True):
            per_replica_losses = strategy.run(train_step, args=(images,gen_training,disc_training,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
        return train_step_dist
    
    train_step_dist=get_train_step_dist()

    @tf.function
    def train_step_dist_ae(images): #training step for autoencoder
        per_replica_losses = strategy.run(train_step_ae, args=(images,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)

    def train_autoencoder(dataset,ae_epochs=AE_EPOCHS):
        print('autoencoder training')
        for epoch in range(ae_epochs):
            for i,images in enumerate(dataset):
                ae_loss=train_step_dist_ae(images)
                if i % 10 ==0:
                    print('\tbatch {} loss {}'.format(i,ae_loss))
            print('epoch: {} ended with ae loss {}'.format(epoch,ae_loss))
    if FID == True:
        iv3_model = InceptionV3(include_top=False, pooling='avg', input_shape=image_dim) # download inception model for FID
        fid_func=calculate_fid(iv3_model)

    def train(dataset,epochs=EPOCHS,picture=True,ae_epochs=AE_EPOCHS,pre_train_epochs=PRE_EPOCHS,name=NAME,save_gen=True,save_disc=True):
        check_dir_gen='./{}/{}/{}'.format(checkpoint_dir,name,'gen')
        check_dir_disc='./{}/{}/{}'.format(checkpoint_dir,name,'disc')
        picture_dir='./{}/{}'.format(gen_img_dir,name)
        for d in [check_dir_gen,check_dir_disc,picture_dir]:
            if not os.path.exists(d):
                os.makedirs(d)
        if AUTO==True:
            print('autoencoder training')
            for epoch in range(ae_epochs):
                for i,images in enumerate(dataset):
                    ae_loss=train_step_dist_ae(images)
                    if i%10==0:
                        print('\tbatch {} autoencoder loss {}'.format(i,ae_loss))
                print('epoch: {} ended with ae loss {}'.format(epoch,ae_loss))
        print('pretraining')
        for epoch in range(pre_train_epochs):
            #dataset.reset()
            for i,images in enumerate(dataset):
                disc_loss,_=train_step_dist(images,gen_training=False)
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
        if AUTO==True:
            intermediate_model=gen.get_layer('autoencoder').get_layer('decoder')
        else:
            intermediate_model=Model(inputs=gen.input, outputs=gen.get_layer('img_output').output)
        interm_noise_dim=intermediate_model.input.shape
        if len(interm_noise_dim)>3:
            interm_noise_dim=interm_noise_dim[1:]
        print('intermediate model loaded')
        for epoch in range(epochs):
            #dataset.reset()
            gen_training=True
            disc_training=True
            for i,images in enumerate(dataset):
                disc_loss,gen_loss=train_step_dist(images,gen_training,disc_training)
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
            print('epoch: {} ended with disc_loss {} and gen loss {} after {} batches'.format(epoch,disc_loss,gen_loss,i))
            if FID == True:
                fid_batch_size=1000
                noise = tf.random.normal([fid_batch_size, * interm_noise_dim])
                gen_imgs=intermediate_model(noise)
                real_imgs=get_real_imgs_fid(art_styles,fid_batch_size)
                fid_score=fid_func(gen_imgs,real_imgs)
                print('fid score = {}'.format(fid_score))
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
                noise = tf.random.normal([1, * interm_noise_dim])
                gen_img=intermediate_model(noise).numpy()
                cv2.imwrite('./{}/{}/epoch_{}.jpg'.format(gen_img_dir,name,epoch),gen_img[0])
    genres=all_genres
    dataset=get_dataset_gen(BLOCK,GLOBAL_BATCH_SIZE,LIMIT,art_styles,genres)
    dataset=strategy.experimental_distribute_dataset(dataset)
    print('main loop')
    #with strategy.scope():
    start=timer()
    train(dataset,EPOCHS,pre_train_epochs=PRE_EPOCHS,name=NAME)
    end=timer()
    print('time elapsed {}'.format(end-start))