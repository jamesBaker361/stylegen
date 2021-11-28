import os
from string_globals import *
#from autoencoderscopy import *

def get_checkpoint_paths(dir):
    '''
    Parameters
    ----------
    dir --str. 
    '''
    return sorted([ f.path for f in os.scandir(dir) if f.is_dir() and str(f).find('epoch') != -1 ],key=lambda x: int(x[x.rfind('_')+1:]),reverse=True)

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
    Returns:
    --------
    most_recent -- str. path of the latest checkpoint
    e -- int. index of latest checkpoint
    
    '''
    ckpt_paths=get_checkpoint_paths(dir)
    return get_ckpt_epoch_from_paths(ckpt_paths)

if __name__=='__main__':
    name='human_flat_block5_conv1'
    check_dir_auto='./{}/{}/{}'.format(checkpoint_dir,name,'auto')
    ckpt_paths=get_checkpoint_paths(check_dir_auto)
    mr,e=get_ckpt_epoch_from_paths(ckpt_paths)
    print(mr,e)
    from autoencoderscopy import *
    model=aegen(block5_conv1,True)
    model.load_weights(mr+'/cp.ckpt')