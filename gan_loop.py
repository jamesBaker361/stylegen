import os
import pdb
from tensorflow.python.keras.models import Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
import argparse
import cv2
from string_globals import *
from other_globals import *


from generator import vqgan,noise_dim_vqgan
from discriminator import conv_discrim

from data_loader import get_dataset

EPOCHS=50
BATCH_SIZE=16
LIMIT=5000

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-2)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-2)

generator=vqgan()
discriminator=conv_discrim()

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    batch_size=images.shape[0]
    noise = tf.random.uniform([batch_size, *noise_dim_vqgan],0,255,dtype=tf.int32)
    #noise = tf.random.normal([batch_size, *noise_dim_vqgan])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    pdb.set_trace()
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return disc_loss,gen_loss

@tf.function
def pre_train_step(images):
    '''this is for pretraining the discriminator
    '''
    batch_size=images.shape[0]
    with tf.GradientTape() as disc_tape:
        generated_images=tf.random.uniform(shape=(3))

def train(dataset,epochs=EPOCHS,picture=True):
    intermediate_model=Model(inputs=generator.input, outputs=generator.get_layer('img_output').output)
    for epoch in range(epochs):
        for i,images in enumerate(dataset):
            disc_loss,gen_loss=train_step(images)
            if i%5==0:
                print('batch {} disc_loss: {} gen loss: {}'.format(i,disc_loss,gen_loss))
        print('epoch: {} ended with disc_loss {} and gen loss {}'.format(epoch,disc_loss,gen_loss))
        if picture:
            noise = tf.random.uniform([1, *noise_dim_vqgan],0,255,dtype=tf.int32)
            gen_img=intermediate_model(noise).numpy()
            cv2.imwrite('epoch_{}.jpg'.format(epoch),gen_img[0])

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='get some args')
    batch_size_str='batch_size'
    epochs_str='epochs'
    limit_str='limit'
    parser.add_argument('--{}'.format(epochs_str),help='foo help',type=int)
    parser.add_argument('--{}'.format(limit_str),help='foo help',type=int)
    parser.add_argument('--{}'.format(batch_size_str),help='foo help',type=int)
    args = parser.parse_args()

    arg_vars=vars(args)
    if batch_size_str in arg_vars:
        BATCH_SIZE=arg_vars[batch_size_str]
    if epochs_str in arg_vars:
        EPOCHS=arg_vars[epochs_str]
    if limit_str in arg_vars:
        LIMIT=arg_vars[limit_str]
    print(BATCH_SIZE,EPOCHS,LIMIT)
    dataset=get_dataset(block1_conv1,BATCH_SIZE,LIMIT)
    train(dataset,EPOCHS)