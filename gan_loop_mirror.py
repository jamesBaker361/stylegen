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
import numpy as np


from generator import vqgen,noise_dim_dcgan,noise_dim_vqgen,dcgen
from autoencoders import aegen

noise_dim=noise_dim_dcgan
from discriminator import conv_discrim

from data_loader import get_dataset_gen, get_dataset_gen_slow,get_real_imgs_fid
from timeit import default_timer as timer
from keras.applications.inception_v3 import InceptionV3
from fid_metric import calculate_fid
from graph_loss import line_graph

EPOCHS=50 #how mnay epochs to train generator for
AE_EPOCHS=50 #how many epochs to pre train autoencoder for
BATCH_SIZE_PER_REPLICA=1
LIMIT=10000 #how many images in total dataset
PRE_EPOCHS=1 #how many epochs to pretrain discriminator on
NAME='testing'
BLOCK=block1_conv1 #which block of vgg we care about
AUTO=True #whether to use autoencoder generator
FID=False #whether to calculate FID score after each epoch
GRAPH_LOSS=True #whether to graph loss of models over time



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

        autoenc=aegen(BLOCK,False)
        gen=autoenc
        disc=conv_discrim(BLOCK)

        noise_dim=gen.input.shape
        if len(noise_dim)>3:
            noise_dim=noise_dim[1:]

    def train_step(images,gen_training,disc_training):
        """A single step to train the generator and discriminator

        Args:
            images: the real genuine images

        Returns:

        """
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
        with tf.GradientTape() as tape:
            reconstructed_images=autoenc(images)
            ae_loss=autoencoder_loss(images,reconstructed_images)

        gradients_of_generator = tape.gradient(ae_loss, autoenc.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, autoenc.trainable_variables))
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
            avg_auto_loss=0.0
            avg_auto_loss_history=[]
            for epoch in range(ae_epochs):
                for i,images in enumerate(dataset):
                    ae_loss=train_step_dist_ae(images)
                    avg_auto_loss+=ae_loss
                    if i%10==0:
                        print('\tbatch {} autoencoder loss {}'.format(i,ae_loss))
                avg_auto_loss=avg_auto_loss/LIMIT
                avg_auto_loss_history.append(avg_auto_loss)
                print('epoch: {} ended with avg ae loss {}'.format(epoch,avg_auto_loss))
                if GRAPH_LOSS==True:
                    line_graph(name,'auto',avg_auto_loss_history)
        avg_disc_loss_history=[]
        avg_gen_loss_history=[]
        if pre_train_epochs>0:
            print('pretraining')
            for epoch in range(pre_train_epochs):
                avg_disc_loss=0.0
                for i,image_tuples in enumerate(dataset):
                    for images in image_tuples:
                        disc_loss,_=train_step_dist(images,gen_training=False,disc_training=True)
                        if i % 10 == 0:
                            print('\tbatch {} disc loss {}'.format(i,disc_loss))
                        avg_disc_loss+=disc_loss/LIMIT
                avg_disc_loss_history.append(avg_disc_loss)
                print('epoch: {} ended with avg_disc_loss {}'.format(epoch,avg_disc_loss))
                if disc_loss<=0.001:
                    print('discriminator converged too quickly')
                    break
                if save_disc is True:
                    save_dir=check_dir_disc+'/pretrain_epoch_{}/'.format(epoch)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    disc.save_weights(save_dir)
        print('training')
        #the intermediate model is used to generate the images
        if AUTO==True:
            intermediate_model=gen.get_layer('autoencoder').get_layer('decoder')
        else:
            intermediate_model=Model(inputs=gen.input, outputs=gen.get_layer('img_output').output)
        interm_noise_dim=intermediate_model.input.shape
        if interm_noise_dim[0]==None:
            interm_noise_dim=interm_noise_dim[1:]
        print('interm_noise_dim ',interm_noise_dim)
        print('intermediate model loaded')
        for epoch in range(epochs):
            gen_training=True
            disc_training=True
            avg_gen_loss=0.0
            avg_disc_loss=0.0
            for i,image_tuples in enumerate(dataset):
                for images in image_tuples: 
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
                    avg_gen_loss+=gen_loss/LIMIT
                    avg_disc_loss+=disc_loss/LIMIT
            avg_disc_loss_history.append(avg_disc_loss)
            avg_gen_loss_history.append(avg_gen_loss)
            print('epoch: {} ended with disc_loss {} and gen loss {} after {} batches'.format(epoch,avg_disc_loss,avg_gen_loss,i))
            if GRAPH_LOSS == True:
                line_graph(name,'generator',avg_gen_loss_history)
                line_graph(name,'discriminator',avg_disc_loss_history)
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
            if picture is True: #creates a 3 x 3 collage of generated images
                for suffix in ['i','ii','iii','iv']:
                    print('suffix ',suffix)
                    collage=[]
                    for x in range(3):
                        row=[]
                        for y in range(3):
                            noise = tf.random.normal([1, *interm_noise_dim])
                            gen_img=intermediate_model(noise).numpy()
                            row.append(gen_img[0])
                        collage.append(cv2.hconcat(row))
                    gen_img_collage=cv2.vconcat(collage)
                    new_img_path='{}/epoch_{}_{}.jpg'.format(picture_dir,epoch,suffix)
                    print('writing ',new_img_path)
                    cv2.imwrite(new_img_path,gen_img_collage)
                    print('the file exists == {}'.format(os.path.exists(new_img_path)))
    genres=[1,7]
    art_styles=[]
    dataset=get_dataset_gen_slow([BLOCK],GLOBAL_BATCH_SIZE,LIMIT,art_styles,genres)
    dataset=strategy.experimental_distribute_dataset(dataset)
    print('main loop')
    print('genres ',genres)
    print('styles ',art_styles)
    print('epochs ',EPOCHS)
    print('ae epochs ', AE_EPOCHS)
    print('dataset size limit ',LIMIT)
    print('discriminator pretraining epochs ',PRE_EPOCHS)
    print('name ',NAME)
    print('block ',BLOCK, SHAPE)
    print('auto? ',AUTO)
    print('test fid? ', FID)
    start=timer()
    train(dataset,EPOCHS,pre_train_epochs=PRE_EPOCHS,name=NAME)
    end=timer()
    print('time elapsed {}'.format(end-start))