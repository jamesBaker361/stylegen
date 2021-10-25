import numpy as np

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
import tensorflow as tf
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from other_globals import *

def calculate_fid(model):
    def _calculate_fid(images1, images2):
        images1 = preprocess_input(images1)
        images2 = preprocess_input(images2)
        # calculate activations
        act1 = model.predict(images1)
        act2 = model.predict(images2)
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = numpy.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
    return _calculate_fid


if __name__ =='__main__':
    model = InceptionV3(include_top=False, pooling='avg', input_shape=image_dim)
    batch_size=1000
    images1=tf.random.normal(shape=(batch_size, * image_dim))
    images2=tf.random.normal(shape=(batch_size, * image_dim))
    fid_metric=calculate_fid(model)
    fid = fid_metric(images1, images2)
    print('FID: %.3f' % fid)