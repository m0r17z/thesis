__author__ = 'moritz'

from os import listdir
from os.path import isfile, join
import cPickle as cp
import time

import warnings

import numpy as np
import theano.tensor as T

from breze.learn.mlp import Mlp
from breze.learn.data import one_hot

import h5py as h5

data = h5.File("/local-home/moritz/PycharmProjects/usaray_learning/usarray_data_scaled_train_val_bin.hdf5", "r")
X = data['trainig_set/train_set'][...]
Z = data['trainig_labels/bin_train_labels'][...]
VX = data['validation_set/val_set'][...]
VZ = data['validation_labels/bin_val_labels'][...]


X = X[:333000]
Z = Z[:333000]
VX = VX[:142000]
VZ = VZ[:142000]
Z = one_hot(Z, 2)
VZ = one_hot(VZ, 2)

'''X = cp.load(open('scaled_samples_train.pkl','r'))
Z = cp.load(open('binary_labels_train.pkl','r'))
VX = cp.load(open('scaled_samples_val.pkl','r'))
VZ = cp.load(open('binary_labels_val.pkl','r'))

X = X[:11500]
Z = Z[:11500]
VX = VX[:5000]
VZ = VZ[:5000]
Z = one_hot(Z, 2)
VZ = one_hot(VZ, 2)'''

input_size = len(X[0])

path = '/local-home/moritz/PycharmProjects/usaray_learning/'

nets = [ f for f in listdir(path) if isfile(join(path,f)) and not f.find('best') ]

best_error = np.inf
best_net = ''

for net in nets:
    file = net
    net = net.replace('.pkl','')
    net = net.replace('best_','')
    net = net.replace('[','')
    net = net.replace(']','')
    net = net.split('_')
    arch = [int(n) for n in net[0].split(',')]
    func = [n.replace(' ','')[1:-1] for n in net[1].split(',')]
    batch_size = int(net[3])
    optimizer = 'gd', {'steprate': 0.1}
    m = Mlp(input_size, arch, 2, hidden_transfers=func, out_transfer='softmax', loss='cat_ce',
            optimizer=optimizer, batch_size=batch_size)
    best_pars = cp.load(open(file,'r'))
    m.parameters.data[...] = best_pars
    n_wrong = 1 - T.eq(T.argmax(m.exprs['output'], axis=1), T.argmax(m.exprs['target'], axis=1)).mean()
    f_n_wrong = m.function(['inpt', 'target'], n_wrong)
    error = f_n_wrong(VX,VZ)
    if error < best_error:
        best_error = error
        best_net = net
    print 'loaded best parameters from file %s' % net
    print 'percentage of misclassified samples on validation/test set: %f' % error

print 'the best net found was ' + str(net) + ' with an error of %f ' % error
