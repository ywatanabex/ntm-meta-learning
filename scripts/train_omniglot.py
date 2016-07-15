import sys
sys.path.append('../')
import argparse
import logging
import pickle
import yaml

import numpy as np
import cupy as cp
import chainer.functions as F
from chainer import optimizers
from chainer import cuda

from utils.generators import OmniglotGenerator
from utils.models import NTM
from utils.metrics import few_shot_score



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1,
                        help='gpu device number. -1 for cpu.')   
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='number of samples in a mini-batch.')   

    # set params
    # -----------
    args = parser.parse_args()        
    batchsize = args.batchsize
    if args.gpu < 0:
        xp = np
    else:
        xp = cp


    # set up training
    # ------------------
    model = NTM(nb_class=5, nb_reads=4, input_size=28*28, 
                cell_size=200, memory_shape=(128, 40), gamma=0.95, 
                gpu=args.gpu)
    optimizer = optimizers.Adam()
    model.set_optimizer(optimizer)

    train_generator = OmniglotGenerator(data_file='../data/omniglot/train.npz', 
                                        nb_classes=5, nb_samples_per_class=10, 
                                        batchsize=batchsize, max_iter=None, xp=xp)
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)

    # start training
    # ----------------
    result = {'Accuracy': {}, 'Loss': {}}
    for i, (images, labels) in train_generator:
        loss = model.train(images, labels)
        logging.info("Episode {}, Train Loss: {}".format(i * batchsize, loss))
        if i % 30 == 0:
            accs = model.evaluate(images, labels)
            accs_ = [cuda.to_cpu(acc) for acc in accs]
            print('-- Accuracies in episodes --')
            print(np.mean(np.array(accs_), axis=1))
        if i % 100 == 0:
            pickle.dump(model, open('model.pkl', 'wb'), protocol=-1)
        if (i != 0) and (i % 500 == 0):
            # evaluate
            # -------------
            print('Evaluation in Test data')
            test_generator = OmniglotGenerator(data_file='../data/omniglot/test.npz', 
                                               nb_classes=5, nb_samples_per_class=10, 
                                               batchsize=256, max_iter=10, xp=xp)
            scores = []
            for i, (images, labels) in test_generator:
                accs = model.evaluate(images, labels)    
                accs_ = [cuda.to_cpu(acc) for acc in accs]
                for b in few_shot_score(accs_, labels):
                    for s in b:
                        scores.append(s)
            result['Accuracy'][i * batchsize] = map(float, np.mean(np.array(scores), axis=0))
            print(('Accuracy: 1st={:.2f}%, 2nd={:.2f}%, 3rd={:.2f}%, 4th={:.2f}%, 5th={:.2f}%, ' +\
                  '6th={:.2f}%, 7th={:.2f}%, 8th={:.2f}%, 9th={:.2f}%, 10th={:.2f}%').format(*100*np.mean(np.array(scores), axis=0)))
            result['Loss'][i * batchsize] = float(loss)
            yaml.dump(result, open('result.yml', 'wb'))


            
      
    
    
    

            



