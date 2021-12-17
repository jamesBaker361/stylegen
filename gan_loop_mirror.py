import os


import tensorflow as tf
print('tf version ='.format(tf. __version__))
import sys
print("Python version")
print (sys.version)

import argparse
import cv2
from helpers import get_checkpoint_paths
from string_globals import *
from other_globals import *

from generator import vqgen,noise_dim_dcgan,noise_dim_vqgen,dcgen
from autoencoderscopy import aegen,extract_generator

noise_dim=noise_dim_dcgan
from discriminator import conv_discrim

from data_loader import get_dataset_gen, get_dataset_gen_slow,get_real_imgs_fid
from timeit import default_timer as timer
from keras.applications.inception_v3 import InceptionV3
from fid_metric import calculate_fid
from graph_loss import data_to_csv
from gif_making import *
from helpers import *
from transfer import *

EPOCHS=50 #how mnay epochs to train generator and discriminator for
AE_EPOCHS=50 #how many epochs to pre train autoencoder for
BATCH_SIZE_PER_REPLICA=1 #batch size per gpu
LIMIT=80000 #how many images in total dataset
PRE_EPOCHS=1 #how many epochs to pretrain discriminator on
NAME='testing'
BLOCK=block1_conv1 #which block of vgg we care about
AUTO=True #whether to use autoencoder generator
FID=False #whether to calculate FID score after each epoch
GRAPH_LOSS=True #whether to graph loss of models over time
FLAT=False #whether to use flat latent or weirdly shaped latent space
HALF=False #whether to just use half the dataset
NO_LOAD=False #whether to load pretrained models 
RESIDUAL=True #whether to use resnext layers in the AEGEN
ATTENTION=True #whether to use attn block layers in the AEGEN



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
    human_str='human'
    baroque_str='baroque'
    renn_str='renn'
    flat_str='flat'
    half_str='half'
    no_load_str='no_load'
    no_attn_str='no_attn'
    no_res_str='no_res'
    parser.add_argument('--{}'.format(epochs_str),help='epochs to train in tandem',type=int)
    parser.add_argument('--{}'.format(limit_str),help='how many images in training set',type=int)
    parser.add_argument('--{}'.format(batch_size_replica_str),help='batch size',type=int)
    parser.add_argument('--{}'.format(pretrain_epochs_str),help='epochs to pre train discriminator',type=int)
    parser.add_argument('--{}'.format(name_str),help='name of this versions', type = str)
    parser.add_argument('--{}'.format(block_str),help='block of vgg we are trying to imitate',type=str)
    parser.add_argument('--{}'.format(auto_str),help='whetther to use an autoencoder GAN',type=bool)
    parser.add_argument('--{}'.format(fid_str),help='whether to calculate fid score after each epoch',type=str)
    parser.add_argument('--{}'.format(human_str),help='only using the human art',type=bool)
    parser.add_argument('--{}'.format(baroque_str),help='only using baroque and romantic styles',type=bool)
    parser.add_argument('--{}'.format(renn_str),help='only using rennaissance styles')
    parser.add_argument('--{}'.format(flat_str),help='flat latent space for noise',type=bool)
    parser.add_argument('--{}'.format(half_str),help='whether to only use half the whole dataset for speed purposes',type=bool)
    parser.add_argument('--{}'.format(no_load_str),help='whether to load past pretrained versions',type=bool)
    parser.add_argument('--{}'.format(no_attn_str),help='whether to not use attentional blocks or not',type=bool)
    parser.add_argument('--{}'.format(no_res_str),help ='whether to not use resnext blocks',type=bool)

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
    if arg_vars[flat_str] is not None:
        FLAT=arg_vars[flat_str]
    if arg_vars[half_str] is not None:
        LIMIT=LIMIT//2
    if arg_vars[no_load_str] is not None:
        NO_LOAD=True
    if arg_vars[no_attn_str] is not None:
        ATTENTION=False
    if arg_vars[no_res_str] is not None:
        RESIDUAL=False
    
    SHAPE=input_shape_dict[BLOCK]

    physical_devices=tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device,True)
        except  RuntimeError as e:
            print(e)

    logical_gpus = tf.config.list_logical_devices('GPU')
    
    print('logical devices: {} physical devices: {}'.format(len(logical_gpus),len(physical_devices)))
    strategy = tf.distribute.MirroredStrategy()

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

        initial_learning_rate = 0.01
        decay_steps = 1000
        decay_rate = 0.9
        learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps, decay_rate)

        autoencoder_optimizer=tf.keras.optimizers.Adam(learning_rate_fn)
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
        discriminator_optimizer = tf.keras.optimizers.SGD()

        output_blocks=[BLOCK]
        autoenc=aegen(BLOCK,flat_latent=FLAT,residual=RESIDUAL,attention=ATTENTION,output_blocks=output_blocks)
        gen=extract_generator(autoenc,BLOCK,output_blocks)
        disc=conv_discrim(BLOCK)

        noise_dim=gen.input.shape
        if len(noise_dim)!=3:
            noise_dim=noise_dim[1:]
        print('noise_dim',noise_dim)
        print('[1, * noise_dim]',[1, * noise_dim])

    def train_step(images,gen_training,disc_training):
        """A single step to train the generator and discriminator

        Args:
            images: the real genuine images

        Returns:

        """
        batch_size=images.shape[0]
        noise = tf.random.uniform([batch_size, * noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = gen(noise, training=gen_training)
            if type(generated_images)==list:
                generated_images=generated_images[0]
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
        '''the entire training function, training autoencoder and GAN

        Parameters:
        ----------

        dataset -- tf BatchedDataSet. The iterable object that supplies all the images
        epochs -- int. how many epochs to train GAN for
        picture -- bool. Whether to make generate images or not
        '''
        check_dir_auto='./{}/{}/{}'.format(checkpoint_dir,name,'auto')
        check_dir_gen='./{}/{}/{}'.format(checkpoint_dir,name,'gen')
        check_dir_disc='./{}/{}/{}'.format(checkpoint_dir,name,'disc')
        picture_dir='./{}/{}'.format(gen_img_dir,name)
        for d in [check_dir_gen,check_dir_disc,picture_dir,check_dir_auto]:
            if not os.path.exists(d):
                os.makedirs(d)
        if AUTO==True:
            print('autoencoder training')
            avg_auto_loss_history=[]
            auto_ckpt_paths=get_checkpoint_paths(check_dir_auto)
            start_epoch=0
            if len(auto_ckpt_paths)>0 and NO_LOAD==False:
                most_recent,start_epoch=get_ckpt_epoch_from_paths(auto_ckpt_paths)
                while 'cp.ckpt.index' not in set(os.listdir(most_recent)):
                    new_start_epoch=max(0,start_epoch-1)
                    second_most_recent=most_recent[:most_recent.rfind('_')+1]+str(new_start_epoch)
                    start_epoch=new_start_epoch
                    most_recent=second_most_recent
                    if start_epoch<=0:
                        break
                if start_epoch>0:
                    autoenc.load_weights(most_recent+'/cp.ckpt')
                    print('successfully loaded autoencoder from epoch {}'.format(start_epoch))
            for epoch in range(start_epoch,ae_epochs,1):
                avg_auto_loss=0.0
                start=timer()
                for i,images in enumerate(dataset):
                    ae_loss=train_step_dist_ae(images)
                    avg_auto_loss+=ae_loss/LIMIT
                    if i%100==0:
                        print('\tbatch {} autoencoder loss {}'.format(i,ae_loss))
                end=timer()
                avg_auto_loss_history.append(avg_auto_loss)
                print('epoch: {} ended with avg ae loss {} time elapsed: {}'.format(epoch,avg_auto_loss,end-start))
                save_dir=check_dir_auto+'/epoch_{}/'.format(epoch)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                autoenc.save_weights(save_dir+'cp.ckpt')
            if GRAPH_LOSS==True and len(avg_auto_loss_history)>0:
                data_to_csv(name,'auto',avg_auto_loss_history)
        avg_disc_loss_history=[]
        avg_gen_loss_history=[]
        disc_ckpt_paths=get_checkpoint_paths(check_dir_disc)
        if pre_train_epochs>0 and len(disc_ckpt_paths)==0:
            print('pretraining')
            for epoch in range(pre_train_epochs):
                avg_disc_loss=0.0
                for i,image_tuples in enumerate(dataset):
                    for images in image_tuples:
                        disc_loss,_=train_step_dist(images,gen_training=False,disc_training=True)
                        if i % 100 == 0:
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
                    disc.save_weights(save_dir+'cp.ckpt')
        print('training')
        #the intermediate model is used to generate the images
        intermediate_model=gen.get_layer('decoder')
        interm_noise_dim=intermediate_model.input.shape
        if interm_noise_dim[0]==None:
            interm_noise_dim=interm_noise_dim[1:]
        print('interm_noise_dim ',interm_noise_dim)
        print('intermediate model loaded')
        if len(disc_ckpt_paths)>0:
            most_recent_disc,start_epoch_disc=get_ckpt_epoch_from_paths(disc_ckpt_paths)
            while 'cp.ckpt.index' not in set(os.listdir(most_recent_disc)):
                new_start_epoch=max(0,start_epoch_disc-1)
                second_most_recent=most_recent_disc[:most_recent_disc.rfind('_')+1]+str(new_start_epoch)
                start_epoch_disc=new_start_epoch
                most_recent_disc=second_most_recent
                if start_epoch_disc<=0:
                    break
            if start_epoch_disc>0:
                disc.load_weights(most_recent_disc+'/cp.ckpt')
                print('successfully loaded discriminator from epoch {}'.format(start_epoch_disc))
        start_epoch_adverse=0
        gen_ckpt_paths=get_checkpoint_paths(check_dir_gen)
        if len(gen_ckpt_paths)>0 and NO_LOAD==False:
            most_recent_gen,start_epoch_adverse=get_ckpt_epoch_from_paths(gen_ckpt_paths)
            while 'cp.ckpt.index' not in set(os.listdir(most_recent_gen)):
                new_start_epoch=max(0,start_epoch_adverse-1)
                second_most_recent=most_recent_gen[:most_recent_gen.rfind('_')+1]+str(start_epoch_adverse)
                start_epoch_adverse=new_start_epoch
                most_recent_gen=second_most_recent
                if start_epoch_adverse<=0:
                    break
            if start_epoch_adverse>0:
                gen.load_weights(most_recent_gen+'/cp.ckpt')
                print('successfully loaded generator from epoch {}'.format(start_epoch_adverse))
        for epoch in range(start_epoch_adverse,epochs,1):
            start=timer() #start a timer to time the epoch
            gen_training=True
            disc_training=True
            avg_gen_loss=0.0
            avg_disc_loss=0.0
            for i,image_tuples in enumerate(dataset):
                for images in image_tuples: 
                    disc_loss,gen_loss=train_step_dist(images,gen_training,disc_training)
                    if i%100==0:
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
            end=timer()
            avg_disc_loss_history.append(avg_disc_loss)
            avg_gen_loss_history.append(avg_gen_loss)
            print('epoch: {} ended with disc_loss {} and gen loss {} after {} batchestime elapsed={}' .format(epoch,avg_disc_loss,avg_gen_loss,i,end-start))
            if GRAPH_LOSS == True:
                data_to_csv(name,'generator',avg_gen_loss_history)
                data_to_csv(name,'discriminator',avg_disc_loss_history)
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
                gen.save_weights(save_dir+'cp.ckpt')
            if save_disc is True:
                save_dir=check_dir_disc+'/epoch_{}/'.format(epoch)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                disc.save_weights(save_dir+'cp.ckpt')
            if picture is True: #creates a generated images
                for suffix in ['i','ii','iii']:
                    '''
                    collage=[]
                    for x in range(3):
                        row=[]
                        for y in range(3):
                            noise = tf.random.normal([1, *interm_noise_dim])
                            gen_img=intermediate_model(noise).numpy()
                            row.append(gen_img[0])
                        collage.append(cv2.hconcat(row))
                    gen_img_collage=cv2.vconcat(collage)
                    '''
                    noise = tf.random.uniform([1, *interm_noise_dim])
                    gen_img=intermediate_model(noise).numpy()[0]
                    new_img_path='{}/epoch_{}_{}.jpg'.format(picture_dir,epoch,suffix)
                    cv2.imwrite(new_img_path,gen_img)
                    print('the file exists == {}'.format(os.path.exists(new_img_path)))
    genres=all_genres_art 
    art_styles=all_styles
    if arg_vars[human_str] is not None:
        genres=[1,3,7,9] #lotta humans 
    elif arg_vars[baroque_str] is not None:
        art_styles=['baroque','romanticism']
    elif arg_vars[renn_str] is not None:
        art_styles=['early-renaissance','high-renaissance','mannerism-late-renaissance','northern-renaissance']
    dataset=get_dataset_gen_slow(output_blocks,GLOBAL_BATCH_SIZE,LIMIT,art_styles,genres)
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
    make_big_gif(NAME) #will save a gif called collage_movie
    print('made big gif')
        