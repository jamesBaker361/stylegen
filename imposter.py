from data_loader import *
from other_globals import *

def imposter(block):
    '''finds .npz files that cause errors and writes a bash script to delete them
    '''
    paths=get_all_img_paths(block)
    print('total images: {}'.format(len(paths)))
    bad=[]
    for img in paths:
        try:
            features=np.load(img)['features']
        except:
            print('attribute error for {}'.format(img))
            bad.append(img)
            continue
        if features.shape != input_shape_dict[block]:
            print(img,features.shape)
            bad.append(img)
    print(len(bad))
    for img in bad:
        splits=img.split('/')
        with open('{}/{}/{}/badboys.sh'.format(npz_root,block1_conv1,splits[2]),'a+') as file:
            file.write('find . -name  \'{}\' -delete\n'.format(splits[3]))
            file.write('rm {}\n'.format(splits[3]))

if __name__=='__main__':
    imposter(block1_conv1)