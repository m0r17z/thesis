# -*- coding: utf-8 -*-

"""Usage: tsne_model.py <data>

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
    tsne = Tsne(n_input, 2)
    print 'TSNE initialized.'
    TX_r = tsne.fit_transform(TX)
    print 'data TSNEd.'

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)
    ax.scatter(TX_r[:, 0], TX_r[:, 1], c=TZ.argmax(axis=1), lw=0, alpha=0.2)
    plt.savefig('/nthome/maugust/thesis/tsne_data.pdf')

if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    print args
    visualize_tsne(args)


