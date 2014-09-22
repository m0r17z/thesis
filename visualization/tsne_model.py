# -*- coding: utf-8 -*-

"""Usage: tsne_model.py <model> <data> <mode>

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
                ax.scatter(trans_TX_r[:, 0], trans_TX_r[:, 1], c=TZ.argmax(axis=1), lw=0, alpha=0.2, s=100)
                plt.savefig('/nthome/maugust/thesis/tsne_model.pdf')

if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    print args
    visualize_tsne(args)


