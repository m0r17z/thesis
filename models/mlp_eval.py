# -*- coding: utf-8 -*-

"""Usage: mlp_eval.py <location> <data>

"""

import docopt
import os
import gzip
import cPickle
import theano.tensor as T
import h5py as h5
import numpy as np

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
                data = h5.File(os.path.join(data_dir,'usarray_data_scaled_train_val_test_real.hdf5'),'r')
                TX = data['test_set/test_set']
                TA = data['test_annotations/test_annotations']
                TZ = data['test_labels/real_test_labels']
                TZ = one_hot(TZ,13)

                n_wrong = 1 - T.eq(T.argmax(trainer.model.exprs['output'], axis=1),
                                   T.argmax(trainer.model.exprs['target'], axis=1)).mean()
                f_n_wrong = trainer.model.function(['inpt', 'target'], n_wrong)

                result = f_n_wrong(TX,TZ)
                result_s = 'model achieved %f%% classification error on the test set' %(result)

                indices = np.random.rand(50) * 90000
                for i in np.arange(50):
                    index = indices[i]
                    prediction = trainer.model.predict(np.reshape(TX[index],(1,len(TX[index]))))
                    if np.argmax(prediction) == np.argmax(TZ[index]):
                        result_s += '\nprediction for sample with annotation %s at position %d is %d with %.3f certainty, correct' \
                                    %(str(TA[index]), index, np.argmax(prediction[0]),prediction[0][np.argmax(prediction[0])])
                    else:
                        result_s += '\nprediction for sample with annotation %s at position %d is %d with %.3f certainty, correct label %d had %.3f certainty' \
                                    %(str(TA[index]), index, np.argmax(prediction[0]),prediction[0][np.argmax(prediction[0])], np.argmax(TZ[index]),prediction[0][np.argmax(TZ[index])])

                print result_s
                with open(os.path.join(dir,'eval_mlp_real_result.txt'),'w') as f:
                    f.write(result_s)

    return 0


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    print args
    evaluate_mlp(args)