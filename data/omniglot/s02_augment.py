"""
Run this script after s01.
"""
import sys
sys.path.append('../../')  
import os
from utils.images import load_transform
from collections import defaultdict

import numpy as np



def load_augment(data_folder, ratio, max_shift, max_rotation, size):
    """
    Args
        data_folder
        ratio (int): data is augmented in this ratio        
        max_shift (int): max shift in original image 
        max_rotation (float): max rotatin in degree
    """
    char_folders = [os.path.join(data_folder, family, character) 
               for family in os.listdir(data_folder) 
               for character in os.listdir(os.path.join(data_folder, family))]

    tmp_dict = defaultdict(list)
    for fld in char_folders:
        print(fld)
        for img in os.listdir(fld):
            # original data
            t_image = load_transform(os.path.join(fld, img), size=size)
            tmp_dict[fld].append(t_image)
            # augmented data
            if ratio != 1:
                angles = np.random.uniform(-max_rotation, max_rotation, size=ratio-1)
                shifts = np.random.randint(-max_shift, max_shift + 1, size=(ratio-1, 2))     
                for angle, shift in zip(angles, shifts):
                    t_image = load_transform(os.path.join(fld, img), 
                                             angle=angle, shift=shift, size=size)
                    tmp_dict[fld].append(t_image)
    results = {key : np.array(value) for key, value in tmp_dict.items()}
    return results


if __name__ == '__main__':    
    print('Convert train dataset')
    data_folder = 'train'
    train_data = load_augment(data_folder, ratio=10, max_shift=5, max_rotation=15.0, size=(28,28))
    np.savez('train.npz', **train_data)

    print('Convert test dataset')    
    data_folder = 'test'
    test_data = load_augment(data_folder, ratio=1, max_shift=0, max_rotation=0, size=(28,28))
    np.savez('test.npz', **test_data)

