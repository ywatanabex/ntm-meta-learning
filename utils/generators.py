"""
Generators of data

This code comes from https://github.com/tristandeleu/ntm-one-shot
"""
import numpy as np
import os
import random
from scipy.misc import imread

from images import load_transform, rotate_right

class OmniglotGenerator(object):
    """OmniglotGenerator

    Args:
        data_file (str): 'data/omniglot/train.npz' or 'data/omniglot/test.npz' 
        nb_classes (int): number of classes in an episode
        nb_samples_per_class (int): nuber of samples per class in an episode
        batchsize (int): number of episodes in each mini batch
        max_iter (int): max number of episode generation
        xp: numpy or cupy
    """
    def __init__(self, data_file, nb_classes=5, nb_samples_per_class=10, 
                 batchsize=64, max_iter=None, xp=np):
        super(OmniglotGenerator, self).__init__()
        self.data_file = data_file
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.batchsize = batchsize
        self.max_iter = max_iter
        self.xp = xp
        self.num_iter = 0
        self.data = self._load_data(self.data_file, self.xp)

    def _load_data(self, data_file, xp):
        data_dict = np.load(data_file)
        return {key: xp.array(val) for (key, val) in data_dict.items()}

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            _images, _labels = zip(*[self.sample(self.nb_classes, self.nb_samples_per_class) 
                           for i in range(self.batchsize)])
            images = [
                self.xp.concatenate(map(lambda x: x.reshape((1,-1)), _img), axis=0)
                for _img in zip(*_images)]
            labels = [_lbl for _lbl in zip(*_labels)]
            return (self.num_iter - 1), (images, labels)
        else:
            raise StopIteration()

    def sample(self, nb_classes, nb_samples_per_class):
        sampled_characters = random.sample(self.data.keys(), nb_classes) # list of keys
        labels_and_images = []
        for (k, char) in enumerate(sampled_characters):
            deg = random.sample(range(4), 1)[0]
            _imgs = self.data[char]
            _ind = random.sample(range(len(_imgs)), nb_samples_per_class)
            labels_and_images.extend([(k, rotate_right(_imgs[i], deg).flatten()) for i in _ind])
            
        random.shuffle(labels_and_images)
        sequence_length = len(labels_and_images)
        labels, images = zip(*labels_and_images)
        return images, labels
