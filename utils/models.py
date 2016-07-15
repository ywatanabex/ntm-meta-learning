import numpy as np
import cupy as cp

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda

from utils.similarities import cosine_similarity

class NTM(object):
    def __init__(self, nb_class, nb_reads, input_size, cell_size, memory_shape, 
                 gamma, gpu=-1):
        """
        Args
            nb_class (int): number of classes in a episode
            nb_reads (int): number of read heads
            input_size (int): dimention of input vector
            cell_size (int): cell size of LSTM controller
            memory_shape (tuple of int): num_memory x dim_memory
            gamma (float) : decay parameter of memory
        """
        self.nb_class = nb_class
        self.nb_reads = nb_reads
        self.input_size = input_size
        self.cell_size = cell_size
        self.memory_shape = memory_shape 
        self.gamma = gamma

        # create chain
        self.chain = self._create_chain()
        self.set_gpu(gpu)


    # Set up methods
    # ---------------
    @property
    def xp(self):
        if self.gpu < 0:
            return np
        else:
            return cp


    def set_gpu(self, gpu):
        self.gpu = gpu
        if self.gpu < 0:
            self.chain.to_cpu()
        else:
            self.chain.to_gpu()


    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.setup(self.chain)


    def _create_chain(self):
        chain = chainer.Chain(
            l_key=L.Linear(self.cell_size, self.nb_reads * self.memory_shape[1]),
            l_add=L.Linear(self.cell_size, self.nb_reads * self.memory_shape[1]),
            l_sigma=L.Linear(self.cell_size, 1),
            l_ho=L.Linear(self.cell_size, self.nb_class),
            l_ro=L.Linear(self.nb_reads * self.memory_shape[1], self.nb_class),
            # for LSTM
            lstm_xh=L.Linear(self.input_size, 4 * self.cell_size),
            lstm_yh=L.EmbedID(self.nb_class, 4 * self.cell_size),
            lstm_rh=L.Linear(self.nb_reads * self.memory_shape[1], 4 * self.cell_size),
            lstm_hh=L.Linear(self.cell_size, 4 * self.cell_size),
            )
        return chain


    # Train methods
    # ---------------
    def make_initial_state(self, batchsize, train=True):
        state = {
            'M': chainer.Variable(
                self.xp.zeros((batchsize,) + self.memory_shape, dtype=self.xp.float32),
                volatile=not train),
            'c': chainer.Variable(
                self.xp.zeros((batchsize, self.cell_size), dtype=self.xp.float32),
                volatile=not train),
            'h': chainer.Variable(
                self.xp.zeros((batchsize, self.cell_size), dtype=self.xp.float32),
                volatile=not train),
            'r': chainer.Variable(self.xp.zeros(
                    (batchsize, self.nb_reads * self.memory_shape[1]), dtype=self.xp.float32),
                                  volatile=not train),
            'read_weight': chainer.Variable(self.xp.zeros(
                    (batchsize, self.nb_reads, self.memory_shape[0]), dtype=self.xp.float32),
                                            volatile=not train),
            'used_weight': chainer.Variable(self.xp.zeros(
                    (batchsize, self.memory_shape[0]), dtype=self.xp.float32),
                                            volatile=not train),
            }
        return state
    

    def forward_onestep(self, x_data, y_data, state, train=True):
        batchsize = x_data.shape[0]
        x = chainer.Variable(x_data, volatile=not train)
        # lstm
        if y_data is not None:
            y = chainer.Variable(self.xp.array(y_data, dtype=self.xp.int32), volatile=not train)
            h_in = self.chain.lstm_xh(x) + \
                self.chain.lstm_yh(y) + \
                self.chain.lstm_rh(state['r']) + \
                self.chain.lstm_hh(state['h'])
        else:
            h_in = self.chain.lstm_xh(x) + \
                self.chain.lstm_rh(state['r']) + \
                self.chain.lstm_hh(state['h'])
            
        c_t, h_t = F.lstm(state['c'], h_in)            

        key = F.reshape(self.chain.l_key(h_t), (batchsize, self.nb_reads, self.memory_shape[1]))
        add = F.reshape(self.chain.l_add(h_t), (batchsize, self.nb_reads, self.memory_shape[1]))
        sigma = self.chain.l_sigma(h_t) 

        # Compute least used weight (not differentiable)
        if self.xp == cp:
            wu_tm1_data = cp.copy(state['used_weight'].data)
            lu_index = np.argsort(cuda.to_cpu(wu_tm1_data), axis=1)[:,:self.nb_reads]
        else:
            wu_tm1_data = state['used_weight'].data
            lu_index = np.argsort(wu_tm1_data, axis=1)[:,:self.nb_reads]
        wlu_tm1_data = self.xp.zeros((batchsize, self.memory_shape[0]), 
                                     dtype=self.xp.float32)
        for i in range(batchsize):
            for  j in range(self.nb_reads):                
                wlu_tm1_data[i,lu_index[i,j]] = 1.  # 1 for least used index
        wlu_tm1 = chainer.Variable(wlu_tm1_data, volatile=not train)

        # write weight
        _wlu_tm1 = F.broadcast_to(
            F.reshape(wlu_tm1, (batchsize, 1, self.memory_shape[0])),
            (batchsize, self.nb_reads, self.memory_shape[0]))
        _sigma = F.broadcast_to(F.reshape(sigma, (batchsize, 1, 1)), 
                                (batchsize, self.nb_reads, self.memory_shape[0]))
        ww_t = _sigma * state['read_weight'] + (1 - _sigma) * _wlu_tm1  
        
        # write to memory
        _lu_mask = 1 - F.broadcast_to(
            F.reshape(wlu_tm1, (batchsize, self.memory_shape[0], 1)),
            (batchsize, self.memory_shape[0], self.memory_shape[1]))
        M_t = state['M'] * _lu_mask + F.batch_matmul(ww_t, add, transa=True) 

        # read from memory
        K_t = cosine_similarity(key, M_t)    

        # read weight, used weight
        wr_t = F.reshape(F.softmax(F.reshape(
                    K_t, (-1, self.memory_shape[0]))), 
                         (batchsize, self.nb_reads, self.memory_shape[0]))
        wu_t = self.gamma * state['used_weight'] + F.sum(wr_t, axis=1) + F.sum(ww_t, axis=1)

        # read memory
        r_t = F.reshape(F.batch_matmul(wr_t, M_t), (batchsize, -1))  # (batchsize, nb_reads * memory_shape[1])

        # set to state
        state_new = {
            'M': M_t,
            'c': c_t,
            'h': h_t,
            'r': r_t,
            'read_weight': wr_t,
            'used_weight': wu_t,
            }
        return state_new

    
    def compute_loss(self, t_data, state, train):
        t = chainer.Variable(self.xp.array(t_data, dtype=self.xp.int32), volatile=not train) 
        u = self.chain.l_ho(state['h']) + self.chain.l_ro(state['r'])
        return F.softmax_cross_entropy(u, t)


    def compute_accuracy(self, t_data, state):
        u = self.chain.l_ho(state['h']) + self.chain.l_ro(state['r'])
        t_est = self.xp.argmax(F.softmax(u).data, axis=1)
        return (t_est == self.xp.array(t_data))


    def train(self, images, labels):
        """
        Train a minibatch of episodes
        """
        batchsize = images[0].shape[0]
        state = self.make_initial_state(batchsize)
        loss = 0
        for x_data, y_data, t_data in zip(images, [None] + labels, labels):
            state = self.forward_onestep(x_data, y_data, state, train=True)
            loss += self.compute_loss(t_data, state, train=True)
        self.optimizer.zero_grads()
        loss.backward()
        self.optimizer.update()
        return loss.data

        
    def evaluate(self, images, labels):
        """
        Evaluate accuracy score
        """
        batchsize = images[0].shape[0]
        state = self.make_initial_state(batchsize, train=False)
        accs = []
        for x_data, y_data, t_data in zip(images, [None] + labels, labels):
            state = self.forward_onestep(x_data, y_data, state, train=False)
            accs.append(self.compute_accuracy(t_data, state))
        return accs


