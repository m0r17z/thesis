# -*- coding: utf-8 -*-

"""Usage: tsne_model.py <data> <output>

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

def visualize_tsne(args):
    data_dir = os.path.abspath(args['<data>'])

    data = h5.File(data_dir,'r')
    TX = data['test_set/test_set'][:5000]
    TZ = data['test_labels/real_test_labels'][:5000]
    TZ = one_hot(TZ,13)
    n_input = TX.shape[1]
    print 'data loaded.'
    tsne = Tsne(n_input, 2, perplexity=5)
    print 'TSNE initialized.'
    TX_r = tsne.fit_transform(TX)
    print 'data TSNEd.'

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)
    TZ_am = TZ.argmax(axis=1)
    ax.scatter(TX_r[TZ_am==0, 0], TX_r[TZ_am==0, 1], c=TZ_am, lw=0, alpha=0.8, s=100, marker='o')
    ax.scatter(TX_r[TZ_am==1, 0], TX_r[TZ_am==1, 1], c=TZ_am, lw=0, alpha=0.8, s=100, marker='v')
    ax.scatter(TX_r[TZ_am==2, 0], TX_r[TZ_am==2, 1], c=TZ_am, lw=0, alpha=0.8, s=100, marker='^')
    ax.scatter(TX_r[TZ_am==3, 0], TX_r[TZ_am==3, 1], c=TZ_am, lw=0, alpha=0.8, s=100, marker='<')
    ax.scatter(TX_r[TZ_am==4, 0], TX_r[TZ_am==4, 1], c=TZ_am, lw=0, alpha=0.8, s=100, marker='>')
    ax.scatter(TX_r[TZ_am==5, 0], TX_r[TZ_am==5, 1], c=TZ_am, lw=0, alpha=0.8, s=100, marker='8')
    ax.scatter(TX_r[TZ_am==6, 0], TX_r[TZ_am==6, 1], c=TZ_am, lw=0, alpha=0.8, s=100, marker='s')
    ax.scatter(TX_r[TZ_am==7, 0], TX_r[TZ_am==7, 1], c=TZ_am, lw=0, alpha=0.8, s=100, marker='p')
    ax.scatter(TX_r[TZ_am==8, 0], TX_r[TZ_am==8, 1], c=TZ_am, lw=0, alpha=0.8, s=100, marker='*')
    ax.scatter(TX_r[TZ_am==9, 0], TX_r[TZ_am==9, 1], c=TZ_am, lw=0, alpha=0.8, s=100, marker='h')
    ax.scatter(TX_r[TZ_am==10, 0], TX_r[TZ_am==10, 1], c=TZ_am, lw=0, alpha=0.8, s=100, marker='H')
    ax.scatter(TX_r[TZ_am==11, 0], TX_r[TZ_am==11, 1], c=TZ_am, lw=0, alpha=0.8, s=100, marker='d')
    ax.scatter(TX_r[TZ_am==12, 0], TX_r[TZ_am==12, 1], c=TZ_am, lw=0, alpha=0.8, s=100, marker='D')
    plt.legend()
    plt.savefig(os.path.join('/nthome/maugust/thesis',args['<output>']))

if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    print args
    visualize_tsne(args)


