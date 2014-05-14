# -*- coding: utf-8 -*-

"""Usage: mlp_eval.py <location> <data>

"""

import docopt
import os
import gzip
import cPickle
import theano.tensor as T
import h5py as h5

from alchemie import contrib
from breze.learn.data import one_hot


def evaluate_mlp(args):
    dir = os.path.abspath(args['<location>'])
    data_dir = os.path.abspath(args['<data>'])
    os.chdir(dir)
    cps = contrib.find_checkpoints('.')

    if cps:
        with gzip.open(cps[-1], 'rb') as fp:
                trainer = cPickle.load(fp)
                trainer.model.parameters.data[...] = trainer.best_pars
                data = h5.File(os.path.join(data_dir,'usarray_data_scaled_train_val_bin.hdf5'),'r')
                VX = data['validation_set/val_set'][...][:140000]
                VZ = data['validation_labels/bin_val_labels'][...][:140000]
                VZ = one_hot(VZ,2)

                n_wrong = 1 - T.eq(T.argmax(trainer.model.exprs['output'], axis=1),
                                   T.argmax(trainer.model.exprs['target'], axis=1)).mean()
                f_n_wrong = trainer.model.function(['inpt', 'target'], n_wrong)

                result = f_n_wrong(VX,VZ)
                result_s = 'model achieved %f%% classification error on the validation set' %(result)
                print result_s
                with open(os.path.join(dir,'eval_mlp_result.txt'),'w') as f:
                    f.write(result_s)
    return 0


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    print args
    evaluate_mlp(args)