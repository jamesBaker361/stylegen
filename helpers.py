import tensorflow as tf
import os
from string_globals import *
#from autoencoderscopy import *

def get_checkpoint_paths(dir):
    '''
    Parameters
    ----------
    dir --str. something like checkpoints/modelname/{gen||auto||disc} 
    '''
    try:
        return sorted([ f.path for f in os.scandir(dir) if f.is_dir() and str(f).find('epoch') != -1 ],key=lambda x: int(x[x.rfind('_')+1:]),reverse=True)
    except FileNotFoundError:
        return []

def get_ckpt_epoch_from_paths(ckpt_paths):
    '''
    Parameters:
    ----------
    ckpt_paths -- [str]. list of paths of checkpoints

    Returns:
    --------
    most_recent -- str. path of the latest checkpoint
    e -- int. index of latest checkpoint
    
    '''
    most_recent=ckpt_paths[0]
    e=int(most_recent[most_recent.rfind('_')+1:])
    return most_recent,e

def get_ckpt_epochs(dir):
    '''
    Parameters:
    -----------
    dir --str. something like checkpoints/modelname/{gen||auto||disc} 

    Returns:
    --------
    most_recent -- str. path of the latest checkpoint
    e -- int. index of latest checkpoint
    
    '''
    ckpt_paths=get_checkpoint_paths(dir)
    return get_ckpt_epoch_from_paths(ckpt_paths)

def _diversity_loss_from_samples(samples):
    '''
    
    Parameters:
    -----------

    samples -- [feature maps].

    Returns:
    -------
    total_loss -- tensor constant. the loss (smaller if images are less similar)
    '''
    batch_size=len(samples)
    total_loss=0.0
    for i in range(batch_size):
        for j in range(i+1,batch_size):
            total_loss+=tf.math.log(tf.norm(samples[i]-samples[j]))
    return -1.0*total_loss

def diversity_loss(gen,batch_size=4):
    ''' the diversity loss is made to penalize the generator for having too similar outputs; based on https://arxiv.org/abs/1701.02096

    Parameters:
    ----------

    gen -- tf.Model. the generator that should make samples
    batch_size -- int. the amount of samples it should make for each step

    Returns:
    -------
    total_loss -- tensor constant. the loss (smaller if images are less similar)

    '''
    gen_noise_dim=gen.input.shape
    if gen_noise_dim[0]==None:
        gen_noise_dim=gen_noise_dim[1:]
    samples=[]
    for _ in range(batch_size):
        noise = tf.random.normal([1, *gen_noise_dim])
        samples.append(gen(noise)[0])
    total_loss=0.0
    for i in range(batch_size):
        for j in range(i+1,batch_size):
            total_loss+=tf.math.log(tf.norm(samples[i]-samples[j]))
    return -1.0*total_loss

if __name__=='__main__':
    name='human_flat_block5_conv1'
    check_dir_auto='./{}/{}/{}'.format(checkpoint_dir,name,'auto')
    ckpt_paths=get_checkpoint_paths(check_dir_auto)
    mr,e=get_ckpt_epoch_from_paths(ckpt_paths)
    print(mr,e)
    from autoencoderscopy import *
    model=aegen(block5_conv1,True)
    model.load_weights(mr+'/cp.ckpt')