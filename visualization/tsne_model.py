# -*- coding: utf-8 -*-

"""Usage: tsne_model.py <model> <data> <mode> <output>

"""

import cPickle
import docopt
import os
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import gzip
import h5py as h5

from breze.learn.tsne import Tsne
from alchemie import contrib
from breze.learn.data import one_hot
import numpy as np
from climin.util import minibatches

def visualize_tsne(args):
    model_dir = os.path.abspath(args['<model>'])
    data_dir = os.path.abspath(args['<data>'])
    os.chdir(model_dir)
    cps = contrib.find_checkpoints('.')

    if cps:
        with gzip.open(cps[-1], 'rb') as fp:
                trainer = cPickle.load(fp)
                trainer.model.parameters.data[...] = trainer.best_pars
                data = h5.File(data_dir,'r')
                TX = data['test_set/test_set'][:5000]
                TZ = data['test_labels/real_test_labels'][:5000]
                TZ = one_hot(TZ,13)
                print 'data loaded.'

                if args['<mode>'] == 'cnn':
                    f_transformed = trainer.model.function(['inpt'],'mlp-layer-2-inpt')
                    print 'transform-function generated.'
                    data = minibatches(TX, trainer.model.batch_size, 0)
                    trans_TX = np.concatenate([f_transformed(element) for element in data], axis=0)
                else:
                    f_transformed = trainer.model.function(['inpt'],'layer-2-inpt')
                    print 'transform-function generated.'
                    trans_TX = f_transformed(TX)

                trans_TX = np.array(trans_TX, dtype=np.float32)
                print 'data transformed'
                trans_n_input = trans_TX.shape[1]
                trans_tsne = Tsne(trans_n_input, 2, perplexity=5)
                print 'TSNE initialized.'
                trans_TX_r = trans_tsne.fit_transform(trans_TX)
                print 'data TSNEd'

                fig = plt.figure(figsize=(16, 16))
                ax = fig.add_subplot(111)
                TZ_am = TZ.argmax(axis=1)
                ax.scatter(trans_TX_r[TZ_am==0, 0], trans_TX_r[TZ_am==0, 1], c='g', lw=0, alpha=1, s=100, marker='o')
                ax.scatter(trans_TX_r[TZ_am==1, 0], trans_TX_r[TZ_am==1, 1], c='b', lw=0, alpha=1, s=100, marker='v')
                ax.scatter(trans_TX_r[TZ_am==2, 0], trans_TX_r[TZ_am==2, 1], c='yellow', lw=0, alpha=1, s=100, marker='^')
                ax.scatter(trans_TX_r[TZ_am==3, 0], trans_TX_r[TZ_am==3, 1], c='r', lw=0, alpha=1, s=100, marker='<')
                ax.scatter(trans_TX_r[TZ_am==4, 0], trans_TX_r[TZ_am==4, 1], c='g', lw=0, alpha=1, s=100, marker='>')
                ax.scatter(trans_TX_r[TZ_am==5, 0], trans_TX_r[TZ_am==5, 1], c='m', lw=0, alpha=1, s=100, marker='8')
                ax.scatter(trans_TX_r[TZ_am==6, 0], trans_TX_r[TZ_am==6, 1], c='crimson', lw=0, alpha=1, s=100, marker='s')
                ax.scatter(trans_TX_r[TZ_am==7, 0], trans_TX_r[TZ_am==7, 1], c='lawngreen', lw=0, alpha=1, s=100, marker='p')
                ax.scatter(trans_TX_r[TZ_am==8, 0], trans_TX_r[TZ_am==8, 1], c='gold', lw=0, alpha=1, s=100, marker='*')
                ax.scatter(trans_TX_r[TZ_am==9, 0], trans_TX_r[TZ_am==9, 1], c='darkorange', lw=0, alpha=1, s=100, marker='h')
                ax.scatter(trans_TX_r[TZ_am==10, 0], trans_TX_r[TZ_am==10, 1], c='k', lw=0, alpha=1, s=100, marker='H')
                ax.scatter(trans_TX_r[TZ_am==11, 0], trans_TX_r[TZ_am==11, 1], c='magenta', lw=0, alpha=1, s=100, marker='d')
                ax.scatter(trans_TX_r[TZ_am==12, 0], trans_TX_r[TZ_am==12, 1], c='turquoise', lw=0, alpha=1, s=100, marker='D')
                plt.savefig(os.path.join('/nthome/maugust/thesis',args['<output>']))

if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    print args
    visualize_tsne(args)


