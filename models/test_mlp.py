__author__ = 'moritz'
import cPickle as cp
import time

import warnings

import numpy as np
import theano.tensor as T

import climin.schedule

import climin.stops
import climin.initialize

from breze.learn.mlp import Mlp
from breze.learn.data import one_hot

import h5py as h5


def run_mlp(arch, func, step, batch, init, X, Z, VX, VZ, wd):

    max_passes = 200
    batch_size = batch
    max_iter = max_passes * X.shape[0] / batch_size
    n_report = X.shape[0] / batch_size

    input_size = len(X[0])

    stop = climin.stops.after_n_iterations(max_iter)
    pause = climin.stops.modulo_n_iterations(n_report)

    #optimizer = 'rmsprop', {'steprate': 0.0001, 'momentum': 0.95, 'decay': 0.8}
    optimizer = 'gd', {'steprate': step}

    m = Mlp(input_size, arch, 2, hidden_transfers=func, out_transfer='softmax', loss='cat_ce',
            optimizer=optimizer, batch_size=batch_size)
    climin.initialize.randomize_normal(m.parameters.data, 0, init)

    losses = []
    print 'max iter', max_iter

    weight_decay = ((m.parameters.in_to_hidden**2).sum()
                    + (m.parameters.hidden_to_out**2).sum()
                    + (m.parameters.hidden_to_hidden_0**2).sum())
    weight_decay /= m.exprs['inpt'].shape[0]
    m.exprs['true_loss'] = m.exprs['loss']
    c_wd = wd
    m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay

    n_wrong = 1 - T.eq(T.argmax(m.exprs['output'], axis=1), T.argmax(m.exprs['target'], axis=1)).mean()
    f_n_wrong = m.function(['inpt', 'target'], n_wrong)

    start = time.time()
    # Set up a nice printout.
    keys = '#', 'seconds', 'loss', 'val loss', 'train emp', 'val emp'
    max_len = max(len(i) for i in keys)
    header = '\t'.join(i for i in keys)
    print header
    print '-' * len(header)
    results = open('results.txt','a')
    results.write(header + '\n')
    results.write('-' * len(header) + '\n')
    results.close()

    for i, info in enumerate(m.powerfit((X, Z), (VX, VZ), stop, pause)):
        if info['n_iter'] % n_report != 0:
            continue
        passed = time.time() - start
        losses.append((info['loss'], info['val_loss']))

        info.update({
            'time': passed,
            'train_emp': f_n_wrong(X, Z),
            'val_emp': f_n_wrong(VX, VZ),
        })

        row = '%(n_iter)i\t%(time)g\t%(loss)g\t%(val_loss)g\t%(train_emp)g\t%(val_emp)g' % info
        results = open('results.txt','a')
        print row
        results.write(row + '\n')
        results.close()

    m.parameters.data[...] = info['best_pars']
    cp.dump(info['best_pars'],open('best_%s_%s_%s_%s_%s.pkl' %(arch,func,step,batch,init),'w'))


if __name__ == '__main__':
    data = h5.File("/local-home/moritz/PycharmProjects/usaray_learning/usarray_data_scaled_train_val_bin.hdf5", "r")
    X = data['trainig_set/train_set'][...]
    Z = data['trainig_labels/bin_train_labels'][...]
    VX = data['validation_set/val_set'][...]
    VZ = data['validation_labels/bin_val_labels'][...]

    print len(X)
    print len(Z)
    print len(VX)
    print len(VZ)

    X = X[:333000]
    Z = Z[:333000]
    VX = VX[:142000]
    VZ = VZ[:142000]
    Z = one_hot(Z, 2)
    VZ = one_hot(VZ, 2)


    results = open('results.txt','w')
    results.close()
    archs = [[100,100],[200,200],[500,500]]
    funcs = [['sigmoid','sigmoid'],['tanh','tanh'],['rectifier','rectifier']]
    steps =[0.1,0.01,0.001,0.0001,0.00001]
    batches = [500,1000,5000,10000]
    inits = [0.1,0.01,0.001,0.0001,0.00001]
    wds = [0.1,0.01,0.001,0.0001,0.00001]

    while 1==1:

        arch_ind = int(np.random.random_sample() * len(archs))
        func_ind = int(np.random.random_sample() * len(funcs))
        step_ind = int(np.random.random_sample() * len(steps))
        batch_ind = int(np.random.random_sample() * len(batches))
        init_ind = int(np.random.random_sample() * len(inits))
        wd_ind = int(np.random.random_sample() * len(wds))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = open('results.txt','a')
            results.write('testing: %s,%s,%s,%s,%s\n' %(archs[arch_ind],funcs[func_ind],steps[step_ind],batches[batch_ind],inits[init_ind]))
            results.close()
            run_mlp(archs[arch_ind],funcs[func_ind],steps[step_ind],batches[batch_ind],inits[init_ind],X,Z,VX,VZ, wds[wd_ind])
