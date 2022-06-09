import os
from keras import backend
import logging
logging.getLogger('tensorflow').disabled = True
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

from data_loader import get_real_imgs_fid,get_dataset_gen_slow_labels
from timeit import default_timer as timer
from keras.applications.inception_v3 import InceptionV3
from fid_metric import calculate_fid
from graph_loss import data_to_csv
from gif_making import *
from helpers import *
from transfer import *
import random
from sklearn.preprocessing import OneHotEncoder

from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.framework.errors_impl import ResourceExhaustedError

#disable_eager_execution()


EPOCHS=2 #how mnay epochs to train generator and discriminator for
AE_EPOCHS=0 #how many epochs to pre train autoencoder for
BATCH_SIZE_PER_REPLICA=2 #batch size per gpu
LIMIT=10 #how many images in total dataset
PRE_EPOCHS=0 #how many epochs to pretrain discriminator on
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
DIVERSITY=True #whether to optimize generator to care about diversity
BETA=0.00025 #beta coefficient on diversity term
CONDITIONAL=False #whether to make it a conditional GAN or not; CGAN uses artistic style labels as input to the flat generator
GAMMA=0.0 #weight for relative weight to put on classification loss- if gamma=0, we wont do classification loss
LOAD_GEN=True #sometimes we want to load the autoencoder but not the generator
OUTPUT_BLOCKS=[BLOCK]
NORM='instance' #what kind of norm to use (batch, group, layer, instande)
ENCODER_NOISE=1.0 #what to multiply encoder noise to
WASSERSTEIN=False #https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
N_CRITIC=5 #how many times to train the discriminator than the generator
GP=False #gradient penalty https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar.py
LAMBDA_GP=10.0


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
    ae_epochs_str='ae_epochs'
    no_diversity_str='no_diversity'
    beta_str='beta'
    conditional_str='conditional'
    gamma_str='gamma'
    output_blocks_str='output_blocks'
    norm_str='norm'
    ukiyo_str='ukiyo'
    encoder_noise_str='encoder_noise'
    wasserstein_str='wasserstein'
    n_critic_str="n_critic"


    parser.add_argument('--{}'.format(epochs_str),help='epochs to train generator/discriminators in tandem (int)',type=int)
    parser.add_argument('--{}'.format(limit_str),help='how many images in training set (int)',type=int)
    parser.add_argument('--{}'.format(batch_size_replica_str),help='batch size (int)',type=int)
    parser.add_argument('--{}'.format(pretrain_epochs_str),help='epochs to pre train discriminator (int)',type=int)
    parser.add_argument('--{}'.format(name_str),help='name of this versions (str)', type = str)
    parser.add_argument('--{}'.format(block_str),help='block of vgg we are training autoencoder with (str)',type=str)
    parser.add_argument('--{}'.format(auto_str),help='whetther to use an autoencoder GAN (bool)',type=bool)
    parser.add_argument('--{}'.format(fid_str),help='whether to calculate fid score after each epoch (bool)',type=str)
    parser.add_argument('--{}'.format(human_str),help='only using the human art (bool)',type=bool)
    parser.add_argument('--{}'.format(baroque_str),help='only using baroque and romantic styles (bool)',type=bool)
    parser.add_argument('--{}'.format(renn_str),help='only using rennaissance styles (bool)', type=bool)
    parser.add_argument('--{}'.format(flat_str),help='flat latent space for noise (bool)',type=bool)
    parser.add_argument('--{}'.format(half_str),help='whether to only use half the whole dataset for speed purposes (bool)',type=bool)
    parser.add_argument('--{}'.format(no_load_str),help='whether to load past pretrained versions of models (bool)',type=bool)
    parser.add_argument('--{}'.format(no_attn_str),help='whether to not use attentional blocks or not (bool)',type=bool)
    parser.add_argument('--{}'.format(no_res_str),help ='whether to not use resnext blocks (bool, deprecated)',type=bool)
    parser.add_argument('--{}'.format(ae_epochs_str),help='how many epochs to train autoencoder for (int)',type=int)
    parser.add_argument('--{}'.format(no_diversity_str),help='whether to train the generator to maximize diversity of samples (bool)',type=bool)
    parser.add_argument('--{}'.format(beta_str),help='beta coefficient on diversity term (float)',type=float)
    parser.add_argument('--{}'.format(conditional_str),help='whether to make it a conditional GAN or not (bool)',type=bool)
    parser.add_argument('--{}'.format(gamma_str),help='gamma coefficient on classification loss (float)',type=float)
    parser.add_argument('--{}'.format(output_blocks_str), nargs='+', default=[])
    parser.add_argument('--{}'.format(norm_str), help='instance batch or group (str)',type=str)
    parser.add_argument('--{}'.format(ukiyo_str),help='whether to only use ukiyo art (bool)',type=bool)
    parser.add_argument('--{}'.format(encoder_noise_str), help='what to multiply the encoder noise by (float)',type=float)
    parser.add_argument('--{}'.format(wasserstein_str),help="whether to use wasserstein GAN architecture",type=bool,default=False)
    parser.add_argument('--{}'.format(n_critic_str),help="how many times to train the discriminator than the generator",type=int,default=5)
    parser.add_argument('--{}'.format("gp"),type=bool,default=False,help="gradient penalty loss")
    parser.add_argument('--{}'.format("lambda_gp"),type=float,default=10.0,help="coefficient on gradient penalty")
    parser.add_argument('--{}'.format("styles"),nargs="+",default=all_styles)
    parser.add_argument('--{}'.format("dc"),type=bool,default=False,help="whetehr to use dcgan architecture")
    parser.add_argument('--{}'.format("base_flat_noise_dim"),type=int,default=64,help="dimensionality of flat latent space")

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
    if arg_vars[ae_epochs_str] is not None:
        AE_EPOCHS=arg_vars[ae_epochs_str]
    if arg_vars[no_diversity_str] is not None:
        DIVERSITY=False
    if arg_vars[beta_str] is not None:
        BETA=arg_vars[beta_str]
    if arg_vars[conditional_str] is not None:
        CONDITIONAL=True
        FLAT=True
    if arg_vars[gamma_str] is not None:
        GAMMA=arg_vars[gamma_str]
    if arg_vars[output_blocks_str] is not None:
        OUTPUT_BLOCKS=arg_vars[output_blocks_str]
    if arg_vars[norm_str] is not None:
        NORM=arg_vars[norm_str].strip()
    if arg_vars[encoder_noise_str] is not None:
        ENCODER_NOISE=arg_vars[encoder_noise_str]
    WASSERSTEIN=args.wasserstein
    N_CRITIC=args.n_critic
    GP=args.gp
    LAMBDA_GP=args.lambda_gp

    if GAMMA!=0:
        CONDITIONAL=True
        FLAT=True
    else:
        CONDITIONAL=False
        
    try:
        OUTPUT_BLOCKS.remove(BLOCK)
    except ValueError:
        pass
    
    OUTPUT_BLOCKS.insert(0,BLOCK)
    
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

    genres=all_genres_art 
    art_styles=args.styles

    #global one_hot
    one_hot=OneHotEncoder()
    one_hot.fit([[s] for s in art_styles])


    with strategy.scope():

        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
        categorical_cross_entropy=tf.keras.losses.CategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
        
        def gram_matrix(input_tensor):
            '''It takes a tensor of shape (1, height, width, channels) and computes the Gram matrix of the channels
            
            Parameters
            ----------
            input_tensor
                The tensor to calculate the gram matrix for.
            
            Returns
            -------
                The gram matrix of the input tensor.
            
            '''
            result = tf.linalg.einsum('ijc,ijd->cd', input_tensor, input_tensor)
            input_shape = tf.shape(input_tensor)
            num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
            return result/(num_locations)

        def discriminator_loss(real_output, fake_output):
            """

            Parameters:
            ----------

            real_output -- tensor. the discriminators predictions from the real images.
            fake_output -- tensor. the discriminators predictions from the fake images.
            """
            real_vector=tf.ones_like(real_output)+tf.random.normal(shape=real_output.shape,mean=0,stddev=.01)
            fake_vector=tf.zeros_like(fake_output)+tf.random.normal(shape=fake_output.shape,mean=0,stddev=.01)
            real_loss = cross_entropy(real_vector, real_output)
            fake_loss = cross_entropy(fake_vector, fake_output)
            total_loss = real_loss + fake_loss
            return tf.nn.compute_average_loss(total_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        def discriminator_loss_single(output,label_number):
            vector=tf.ones_like(output) #+tf.random.normal(shape=tf.shape(output),mean=0,stddev=.1)
            vector +=tf.random.normal(shape=tf.shape(vector),mean=0,stddev=.1)
            vector=label_number* vector
            loss=cross_entropy(vector,output)
            return tf.nn.compute_average_loss(loss, global_batch_size=GLOBAL_BATCH_SIZE)

        def wasserstein_loss(real_labels, pred_labels):
            return backend.mean(real_labels*pred_labels)

        def gradient_penalty_loss(fake_data,real_data,disc):
            alpha=tf.constant(random.random())
            fake_data=list(fake_data)
            differences = [f-r for f,r in zip(fake_data,real_data)]#fake_data - real_data
            interpolates = [r + (alpha*d) for r,d in zip(real_data,differences)]
            if len(fake_data)==1:
                gradients = tf.gradients(disc(interpolates[0][0])[0], interpolates)
            else:
                gradients = tf.gradients(disc(interpolates)[0], interpolates)
            gradient_penalty=0.0
            for g in gradients:
                slopes = tf.sqrt(tf.reduce_sum(tf.square(g)))
                gradient_penalty += tf.reduce_mean((slopes-1.)**2)
            return tf.nn.compute_average_loss([gradient_penalty], global_batch_size=GLOBAL_BATCH_SIZE)


        def generator_loss(fake_output):
            """

            Parameters:
            -----------

            fake_output -- tensor. the discriminators predictions from the fake images.
            """
            vector=tf.ones_like(fake_output)+tf.random.normal(shape=fake_output.shape,mean=0,stddev=.01)
            loss=cross_entropy(vector, fake_output)
            return tf.nn.compute_average_loss(loss, global_batch_size=GLOBAL_BATCH_SIZE)

        def diversity_loss_from_samples(samples):
            '''the diversity loss is made to penalize the generator for having too similar outputs; based on https://arxiv.org/abs/1701.02096
            
            Parameters:
            -----------

            samples -- [feature maps].

            Returns:
            -------
            total_loss -- tensor constant. the loss (approaches -infinity when images are less similar)
            '''
            batch_size=GLOBAL_BATCH_SIZE #len(samples)
            loss=0 #[-1.0* tf.reduce_mean(tf.square(tf.subtract(samples, samples)))]
            gram_samples=[gram_matrix(s) for s in samples]
            for i in range(batch_size):
                for j in range(i+1,batch_size):
                    loss+=tf.norm(gram_samples[i]-gram_samples[j])
            return tf.nn.compute_average_loss([-loss], global_batch_size=GLOBAL_BATCH_SIZE)
        
        def diversity_loss_from_samples_and_noise(samples,noise):
            '''based off of https://arxiv.org/abs/1901.09024
            '''
            try:
                batch_size=len(samples)
            except TypeError:
                batch_size=samples.shape[0]
            loss=tf.constant(0.0,dtype=tf.float32) #[-1.0* tf.reduce_mean(tf.square(tf.subtract(samples, samples)))]
            for i in range(batch_size):
                for j in range(i+1,batch_size):
                    loss+=tf.norm(samples[i]-samples[j])/tf.norm(noise[i]-noise[j])
            return tf.nn.compute_average_loss([-loss], global_batch_size=GLOBAL_BATCH_SIZE)

        def classification_loss(labels,predicted_labels):
            """computes loss between real and predicted style labels

            """
            loss=categorical_cross_entropy(labels,predicted_labels)
            return tf.nn.compute_average_loss(loss, global_batch_size=GLOBAL_BATCH_SIZE)

        def autoencoder_loss(images, generated_images):
            loss=[tf.reduce_mean(tf.square(tf.subtract(images, generated_images)))]
            return tf.nn.compute_average_loss(loss,global_batch_size=GLOBAL_BATCH_SIZE)

        initial_learning_rate = 0.01
        decay_steps = 50000
        decay_rate = 0.9
        learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps, decay_rate)

        autoencoder_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005)
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=0.5)
        if WASSERSTEIN or GP:
            discriminator_optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00005)
        else:
            discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, beta_1=0.5)

        flat_latent_dim=0
        if FLAT==True:
            flat_latent_dim=args.base_flat_noise_dim
        if CONDITIONAL == True and args.dc==False:
            flat_latent_dim+=len(art_styles)
            autoenc=aegen(BLOCK,base_flat_noise_dim=args.base_flat_noise_dim,residual=RESIDUAL,attention=ATTENTION,art_styles=art_styles,output_blocks=[BLOCK],norm=NORM)
        elif args.dc and CONDITIONAL==False:
            autoenc=aegen(BLOCK,base_flat_noise_dim=args.base_flat_noise_dim,residual=RESIDUAL,attention=ATTENTION,output_blocks=[BLOCK],norm=NORM,dc_enc=True,dc_dec=True)
        elif args.dc and CONDITIONAL:
            autoenc=aegen(BLOCK,base_flat_noise_dim=args.base_flat_noise_dim,residual=RESIDUAL,attention=ATTENTION,art_styles=art_styles,output_blocks=[BLOCK],norm=NORM,dc_enc=True,dc_dec=True)
        else:
            autoenc=aegen(BLOCK,residual=RESIDUAL,attention=ATTENTION,output_blocks=[BLOCK],norm=NORM)
        gen=extract_generator(autoenc,BLOCK,OUTPUT_BLOCKS)
        discs=[conv_discrim(b,len(art_styles),wasserstein=WASSERSTEIN,gp=GP) for b in OUTPUT_BLOCKS]
        truth_value=tf.concat([d.output[0] for d in discs],-1)
        classification_value=tf.concat([d.output[1] for d in discs],-1)
        mega_disc=tf.keras.Model(inputs=[d.input for d in discs],outputs=[truth_value,classification_value])
        print(mega_disc.outputs)

        noise_dim=gen.input.shape
        if FLAT==False and len(noise_dim)!=3:
            noise_dim=noise_dim[1:]
        if FLAT==True and len(noise_dim)!=1:
            noise_dim=noise_dim[1:]
        gen.build(noise_dim)
        print('noise_dim',noise_dim)
        print('[1, * noise_dim]',[1, * noise_dim])

    @tf.function
    def train_step_diversity(diversity_batch_size):
        diversity_loss_list=[]
        with tf.GradientTape(persistent=True) as gen_tape:
            sample_noise=tf.random.normal([diversity_batch_size, * noise_dim])
            diversity_generated_samples=gen(sample_noise)
            if len(discs)==1:
                diversity_generated_samples=[diversity_generated_samples]
            for samples in diversity_generated_samples:
                _div_loss=diversity_loss_from_samples_and_noise(samples,sample_noise)/diversity_batch_size
                _div_loss*=BETA
                diversity_loss_list.append(_div_loss)
        for loss in diversity_loss_list:
            gradients_of_generator = gen_tape.gradient(loss, gen.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
        return sum(diversity_loss_list)


    @tf.function
    def train_step(images,gen_training,disc_training):
        """A single step to train the generator and discriminator

        Parameters:
        -----------
        images -- []. the real genuine images
        gen_training -- bool. whether to train the generator this step.
        disc_training -- bool. whether to train the discriminator this step.
        diversity_training -- bool. whether to train for diversity
        

        Returns:
        --------
        disc_loss -- float. discriminator loss
        gen_loss -- float. generator loss
        div_loss. float. generator diversity loss
        """
        labels=images[-1]
        authentic_images=images[:-1]
        batch_size=tf.shape(images[0])[0]
        
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:

            if CONDITIONAL == True:
                generic_noise=tf.random.normal([batch_size,args.base_flat_noise_dim])
                art_style_encoding_list=tf.random.uniform([batch_size,len(art_styles)],minval=0,maxval=1)
                noise=tf.concat([generic_noise,art_style_encoding_list],axis=-1)
            else:
                noise = tf.random.normal([batch_size, * noise_dim],dtype=tf.float64)
                
            generated_images=gen(noise, training=gen_training)

            real_output,real_labels = mega_disc(authentic_images, training=disc_training) 
            fake_output,fake_labels = mega_disc(generated_images, training=disc_training)
            
                
            if WASSERSTEIN or GP:
                disc_loss_real=wasserstein_loss(real_output,tf.ones_like(real_output)+tf.random.normal(shape=tf.shape(real_output),mean=0,stddev=.01))
                disc_loss_fake=wasserstein_loss(fake_output,-tf.ones_like(real_output)-tf.random.normal(shape=tf.shape(real_output),mean=0,stddev=.01))
            else:
                disc_loss_real = discriminator_loss_single(real_output,1)
                disc_loss_fake =discriminator_loss_single(fake_output,0)

            _disc_loss= disc_loss_fake+disc_loss_real

            if GP:
                if len(discs)>1:
                    gp=gradient_penalty_loss(authentic_images,generated_images,mega_disc)
                else:
                    gp=gradient_penalty_loss([authentic_images],[generated_images],mega_disc)
                _disc_loss+=gp

            if gen_training == True:
                _gen_loss= generator_loss(fake_output)
            else:
                _gen_loss=tf.constant(0.0)

            if CONDITIONAL and GAMMA !=0:
                gen_class_label_loss= GAMMA * classification_loss(fake_labels, tf.concat([art_style_encoding_list for _ in discs],axis=-1))
                disc_class_label_loss= GAMMA * classification_loss(real_labels,tf.concat([labels for _ in discs],axis=-1))
                class_label_loss= gen_class_label_loss+disc_class_label_loss
                if gen_training:
                    _gen_loss+=gen_class_label_loss
                if disc_training:
                    _disc_loss+=disc_class_label_loss
            else:
                class_label_loss=tf.constant(0.0)

            #disc_loss+=disc_class_label_loss
            #combined_loss_sum=sum(combined_loss_list)
        if gen_training is True:
            gradients_of_generator = gen_tape.gradient(_gen_loss, gen.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
            del gradients_of_generator
        
        if disc_training is True:
            gradients_of_discriminator = disc_tape.gradient(_disc_loss, mega_disc.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, mega_disc.trainable_variables))
            del gradients_of_discriminator

        del disc_tape
        del gen_tape
        #print(disc_loss_list)
        
        #return tf.add_n([tf.add_n(dl) for dl in disc_loss_list]),0,0,0
        #return strategy.experimental_local_results(0),strategy.experimental_local_results(0),strategy.experimental_local_results(0),strategy.experimental_local_results(0)
        return _gen_loss,_disc_loss,class_label_loss
        #return 0,0,0,0
        #return tf.add_n([tf.add_n(dl) for dl in disc_loss_list]),tf.add_n(combined_loss_list),tf.add_n(diversity_loss_list),tf.add_n(class_label_loss_list)

    @tf.function
    def train_step_ae(images): #training autoencoder to reconstruct things, not generate
        with tf.GradientTape() as tape:
            labels=images[-1]
            images=images[0]
            #print(len(images))
            if CONDITIONAL==False:
                reconstructed_images=autoenc(images) #if its not conditional, then the AE doesn't need the labels
            else:
                reconstructed_images=autoenc((images,labels))
            ae_loss=autoencoder_loss(images,reconstructed_images)

        gradients_of_generator = tape.gradient(ae_loss, autoenc.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, autoenc.trainable_variables))
        return ae_loss


    #def get_train_step_dist():
    @tf.function
    def train_step_dist(images,gen_training=True,disc_training=True):
        per_replica_losses = strategy.run(train_step, args=(images,gen_training,disc_training,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
    #return train_step_dist
    
    #train_step_dist=get_train_step_dist()

    @tf.function
    def train_step_dist_ae(images): #training step for autoencoder
        per_replica_losses = strategy.run(train_step_ae, args=(images,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)

    @tf.function
    def train_step_dist_div(diversity_batch_size=4):
        per_replica_losses = strategy.run(train_step_diversity, args=(diversity_batch_size,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)

    if FID == True:
        iv3_model = InceptionV3(include_top=False, pooling='avg', input_shape=image_dim) # download inception model for FID
        fid_func=calculate_fid(iv3_model)

    #@tf.function
    def train(dataset,epochs=EPOCHS,picture=True,ae_epochs=AE_EPOCHS,pre_train_epochs=PRE_EPOCHS,name=NAME,one_hot=one_hot,save_gen=True,save_disc=True,n_critic=1):
        check_dir_auto='./{}/{}/{}'.format(checkpoint_dir,name,'auto')
        check_dir_gen='./{}/{}/{}'.format(checkpoint_dir,name,'gen')
        check_dir_disc_list=['./{}/{}/{}/{}'.format(checkpoint_dir,name,'disc',b) for b in OUTPUT_BLOCKS]
        picture_dir='./{}/{}'.format(gen_img_dir,name)
        for d in [check_dir_gen,picture_dir,check_dir_auto]:
            if not os.path.exists(d):
                os.makedirs(d)
        if AUTO==True:
            print('autoencoder training')
            avg_auto_loss_history=[]
            auto_ckpt_paths=get_checkpoint_paths(check_dir_auto)
            start_epoch=0
            if len(auto_ckpt_paths)>0 and NO_LOAD==False:
                most_recent,start_epoch=get_ckpt_epoch_from_paths(auto_ckpt_paths)
                while 'cp.ckpt.index' not in set(os.listdir(most_recent)): #there are some improperly saved models so we have to find the right ones
                    new_start_epoch=max(0,start_epoch-1)
                    second_most_recent=most_recent[:most_recent.rfind('_')+1]+str(new_start_epoch)
                    start_epoch=new_start_epoch
                    most_recent=second_most_recent
                    if start_epoch<=0:
                        break
                if start_epoch>0:
                    #autoenc.load_weights(most_recent+'/cp.ckpt')
                    print('successfully loaded autoencoder from epoch {}'.format(start_epoch))
                    start_epoch+=1
                    if ae_epochs>start_epoch:
                        LOAD_GEN=False
            for epoch in range(start_epoch,ae_epochs,1):
                avg_auto_loss=0.0
                start=timer()
                i=0
                for images in dataset:
                    try:
                        ae_loss=train_step_dist_ae(images)
                    except ResourceExhaustedError:
                        print("OOM! batch_size per replica: {} len blocks: {} base_flat_noise_dim: {}".format(BATCH_SIZE_PER_REPLICA, len(discs), args.base_flat_noise_dim))
                        exit()
                    avg_auto_loss+=ae_loss/LIMIT
                    if i%100==0:
                        print('\tbatch {} autoencoder loss {}'.format(i,ae_loss))
                    i+=1
                end=timer()
                avg_auto_loss_history.append(avg_auto_loss)
                print('epoch: {} ended with avg ae loss {} time elapsed: {}'.format(epoch,avg_auto_loss,end-start))
                save_dir=check_dir_auto+'/epoch_{}/'.format(epoch)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                #autoenc.save_weights(save_dir+'cp.ckpt')
            if GRAPH_LOSS==True and len(avg_auto_loss_history)>0:
                data_to_csv(name,'auto',avg_auto_loss_history)
        avg_disc_loss_history=[]
        avg_gen_loss_history=[]
        avg_diversity_loss_history=[]
        check_dir_disc=check_dir_disc_list[0]
        disc_ckpt_paths=get_checkpoint_paths(check_dir_disc)
        if pre_train_epochs>0 and (len(disc_ckpt_paths)==0 or NO_LOAD==True):
            print('pretraining')
            for epoch in range(pre_train_epochs):
                avg_disc_loss=0.0
                i=0
                for images in dataset:
                    #for images in image_tuples:
                    try:
                        disc_loss,_,__=train_step_dist(images,gen_training=False,disc_training=True)
                    except ResourceExhaustedError:
                        print("OOM! batch_size per replica: {} len blocks: {} base_flat_noise_dim: {}".format(BATCH_SIZE_PER_REPLICA, len(discs), args.base_flat_noise_dim))
                        exit()
                    if i % 100 == 0:
                        print('\tbatch {} disc loss {}'.format(i,disc_loss))
                    avg_disc_loss+=disc_loss/LIMIT
                    i+=1
                avg_disc_loss_history.append(avg_disc_loss)
                print('epoch: {} ended with avg_disc_loss {}'.format(epoch,avg_disc_loss))
                for disc,check_dir_disc in zip(discs,check_dir_disc_list):
                    disc_ckpt_paths=get_checkpoint_paths(check_dir_disc)
                    if pre_train_epochs>0 and len(disc_ckpt_paths)<pre_train_epochs:
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
        for disc,check_dir_disc in zip(discs,check_dir_disc_list):
            disc_ckpt_paths=get_checkpoint_paths(check_dir_disc)
            if len(disc_ckpt_paths)>0:
                most_recent_disc,start_epoch_disc=get_ckpt_epoch_from_paths(disc_ckpt_paths)
                while 'cp.ckpt.index' not in set(os.listdir(most_recent_disc)):
                    new_start_epoch=max(0,start_epoch_disc-1)
                    second_most_recent=most_recent_disc[:most_recent_disc.rfind('_')+1]+str(new_start_epoch)
                    start_epoch_disc=new_start_epoch
                    most_recent_disc=second_most_recent
                    if start_epoch_disc<=0:
                        break
                if start_epoch_disc>0 and NO_LOAD==False:
                    #disc.load_weights(most_recent_disc+'/cp.ckpt')
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
            if start_epoch_adverse>0 and NO_LOAD==False:
                #gen.load_weights(most_recent_gen+'/cp.ckpt')
                print('successfully loaded generator from epoch {}'.format(start_epoch_adverse))
                start_epoch_adverse+=1
        for epoch in range(start_epoch_adverse,epochs,1):
            start=timer() #start a timer to time the epoch
            gen_training=True
            disc_training=True
            diversity_training=DIVERSITY
            avg_gen_loss=0.0
            avg_disc_loss=0.0
            avg_diversity_loss=0.0
            avg_class_loss=0.0
            i=0
            for images in dataset:
                #for images in image_tuples:
                if i% n_critic ==0:
                    gen_training=True
                    diversity_training=DIVERSITY
                else:
                    gen_training=False
                    diversity_training=False
                try:
                    disc_loss,gen_loss,class_label_loss=train_step_dist(images,gen_training,disc_training)
                except ResourceExhaustedError:
                    print("OOM! batch_size per replica: {} len blocks: {} base_flat_noise_dim: {}".format(BATCH_SIZE_PER_REPLICA, len(discs), args.base_flat_noise_dim))
                    exit()
                if diversity_training:
                    div_loss=train_step_dist_div()
                else:
                    div_loss=0.0
                if i%100==0: #print out the loss every 100 batches
                    print('\tbatch {} disc_loss: {} gen loss: {} diversity loss: {} class loss: {}'.format(i,disc_loss,gen_loss,div_loss,class_label_loss))
                i+=1
            end=timer()
            print('epoch: {} ended with disc_loss {} gen loss {} div_loss {} class lost {} after {} batches time elapsed={}' .format(epoch,disc_loss,gen_loss,div_loss,class_label_loss,i,end-start))
            if GRAPH_LOSS == True:
                data_to_csv(name,'generator',avg_gen_loss_history)
                data_to_csv(name,'discriminator',avg_disc_loss_history)
                data_to_csv(name,'diversity',avg_diversity_loss_history)
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
                #gen.save_weights(save_dir+'cp.ckpt')
            if save_disc is True:
                save_dir=check_dir_disc+'/epoch_{}/'.format(epoch)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                #disc.save_weights(save_dir+'cp.ckpt')
            if picture is True: #creates generated images
                for suffix in ['i','ii']:
                    noise = tf.random.normal([1, *interm_noise_dim])
                    #gen_img=intermediate_model(noise)
                    gen_img=intermediate_model(noise).numpy()[0]
                    new_img_path='{}/epoch_{}_{}.jpg'.format(picture_dir,epoch,suffix)
                    #image_tensors.append(gen_img)
                    #image_paths.append(new_img_path)
                    cv2.imwrite(new_img_path,gen_img)
                    #print('the file exists == {}'.format(os.path.exists(new_img_path)))
    
    dataset=get_dataset_gen_slow_labels(OUTPUT_BLOCKS,GLOBAL_BATCH_SIZE,one_hot,LIMIT,art_styles,genres)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    #dataset=get_dataset_gen_slow(OUTPUT_BLOCKS,GLOBAL_BATCH_SIZE,LIMIT,art_styles,genres)
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
    print("eagerly = ",tf.executing_eagerly())
    start=timer()
    if WASSERSTEIN:
        train(dataset,EPOCHS,pre_train_epochs=PRE_EPOCHS,name=NAME,n_critic=N_CRITIC)
    else:
        train(dataset,EPOCHS,pre_train_epochs=PRE_EPOCHS,name=NAME)
    end=timer()
    print('time elapsed {}'.format(end-start))
    make_big_gif(NAME) #will save a gif called collage_movie
    print('made big gif')
        