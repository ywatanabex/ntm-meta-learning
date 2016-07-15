"""
Image utility functions

This code comes from https://github.com/tristandeleu/ntm-one-shot
"""
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import shift as scipy_shift
import scipy.misc
from scipy.misc import imread, imresize

import os
import random


def time_offset_input(labels_and_images):
    labels, images = zip(*labels_and_images)
    time_offset_labels = (None,) + labels[:-1]
    return zip(images, time_offset_labels)

def load_transform(image_path, angle=0., shift=(0, 0), size=(20, 20)):
    # Load the image
    original = imread(image_path, flatten=True)
    original /= np.max(original)
    # Rotate the image
    rotated = np.maximum(np.minimum(rotate(original, angle=angle, cval=1.), 1.), 0.)
    # Shift the image
    shifted = scipy_shift(rotated, shift=shift, cval=1.)
    # Resize the image
    resized = np.asarray(scipy.misc.imresize(rotated, size=size), dtype=np.float32) / 255.
    # Invert the image
    inverted = 1. - resized
    max_value = np.max(inverted)
    if max_value > 0.:
        inverted /= max_value
    return inverted

def rotate_right(img, angle):    
    if angle == 0:
        return img  
    elif angle == 1:  # 90 degree
        return img.T[:,::-1]
    elif angle == 2:  # 180 degree
        return img[::-1,::-1]
    elif angle == 3:  # 270 degree
        return img.T[::-1,:]
    else:
        raise ValueError('angle must be 0, 1, 2 or 3')

        
