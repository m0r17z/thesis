# -*- coding: utf-8 -*-

"""Usage: eval_model.py <location> <data> <mode>

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
    data = os.path.abspath(args['<data>'])
    mode = args['<mode>']
    os.chdir(dir)
    cps = contrib.find_checkpoints('.')

    if cps:
        with gzip.open(cps[-1], 'rb') as fp:
            trainer = cPickle.load(fp)
            trainer.model.parameters.data[...] = trainer.best_pars
            cPickle.dump(trainer.best_pars, open('best_pars.pkl','wb'))
            data = h5.File(data,'r')
            TX = data['test_set/test_set']
            TA = data['test_annotations/test_annotations']
            TZ = data['test_labels/real_test_labels']
            TZ = one_hot(TZ,13)

            n_wrong = 1 - T.eq(T.argmax(trainer.model.exprs['output'], axis=1),
                               T.argmax(trainer.model.exprs['target'], axis=1)).mean()
            f_n_wrong = trainer.model.function(['inpt', 'target'], n_wrong)

            f_pos = T.mean(T.neq(T.argmax(trainer.model.exprs['output'], axis=1),0) * T.eq(T.argmax(trainer.model.exprs['target'], axis=1), 0))
            f_f_pos = trainer.model.function(['inpt', 'target'], f_pos)

            f_neg = T.mean(T.eq(T.argmax(trainer.model.exprs['output'], axis=1),0) * T.neq(T.argmax(trainer.model.exprs['target'], axis=1), 0))
            f_f_neg = trainer.model.function(['inpt', 'target'], f_neg)

            if mode == 'cnn':
                print 'using cnn model'
                emp_loss = trainer.model.apply_minibatches_function(f_n_wrong,TX,TZ)
                f_p = trainer.model.apply_minibatches_function(f_f_pos,TX,TZ)
                f_n = trainer.model.apply_minibatches_function(f_f_neg,TX,TZ)
            else:
                emp_loss = f_n_wrong(TX,TZ)
                f_p = f_f_pos(TX,TZ)
                f_n = f_f_neg(TX,TZ)

            P_pos = np.argmax(trainer.model.predict(TX),axis=1)
            Z_pos = np.argmax(TZ, axis=1)

            neighbour_fails = .0
            relevant_fails = 0

            for i in np.arange(len(P_pos)):
                if P_pos[i] > 0 and Z_pos[i] > 0 and P_pos[i] != Z_pos[i]:
                    relevant_fails += 1
                    if is_neighbour(P_pos[i],Z_pos[i]):
                        neighbour_fails += 1

            if not relevant_fails == 0:
		neighbour_fails /= relevant_fails


            emp_loss_s = 'model achieved %f%% classification error on the test set' %emp_loss
            f_p_s = '\nmodel achieved %f%% false positives on the test set' %f_p
            f_n_s = '\nmodel achieved %f%% false negatives on the test set' %f_n
            neigh_s = '\nmodel achieved %f%% neighbour misspredictions on the test set' %neighbour_fails

            print emp_loss_s
            print f_p_s
            print f_n_s
            print neigh_s
            with open(os.path.join(dir,'eval_result.txt'),'w') as f:
                f.write(emp_loss_s)
                f.write(f_p_s)
                f.write(f_n_s)
                f.write(neigh_s)

    return 0

    '''indices = np.random.rand(50) * 10000
     for i in np.arange(50):
         index = indices[i]
         prediction = trainer.model.predict(np.reshape(TX[index],(1,len(TX[index]))))
         if np.argmax(prediction) == np.argmax(TZ[index]):
             result_s += '\nCORRECT: prediction for sample with annotation %s at position %d is %d with %.3f certainty' \
                         %(str(TA[index]), index, np.argmax(prediction[0]),prediction[0][np.argmax(prediction[0])])
         else:
             result_s += '\nWRONG: prediction for sample with annotation %s at position %d is %d with %.3f certainty, correct label %d had %.3f certainty' \
                         %(str(TA[index]), index, np.argmax(prediction[0]),prediction[0][np.argmax(prediction[0])], np.argmax(TZ[index]),prediction[0][np.argmax(TZ[index])])'''

def is_neighbour(a,b):
    # assumption: a != b and a != 0 and b != 0
    if a == 1:
        if b == 2:
            return True
        if b == 3:
            return False
        if b == 4:
            return False
        if b == 5:
            return True
        if b == 6:
            return True
        if b == 7:
            return False
        if b == 8:
            return False
        if b == 9:
            return False
        if b == 10:
            return False
        if b == 11:
            return False
        if b == 12:
            return False
        else:
            return False
    if a == 2:
        if b == 1:
            return True
        if b == 3:
            return True
        if b == 4:
            return False
        if b == 5:
            return True
        if b == 6:
            return True
        if b == 7:
            return True
        if b == 8:
            return False
        if b == 9:
            return False
        if b == 10:
            return False
        if b == 11:
            return False
        if b == 12:
            return False
        else:
            return False
    if a == 3:
        if b == 1:
            return False
        if b == 2:
            return True
        if b == 4:
            return True
        if b == 5:
            return False
        if b == 6:
            return True
        if b == 7:
            return True
        if b == 8:
            return True
        if b == 9:
            return False
        if b == 10:
            return False
        if b == 11:
            return False
        if b == 12:
            return False
        else:
            return False
    if a == 4:
        if b == 1:
            return False
        if b == 2:
            return False
        if b == 3:
            return True
        if b == 5:
            return False
        if b == 6:
            return False
        if b == 7:
            return True
        if b == 8:
            return True
        if b == 9:
            return False
        if b == 10:
            return False
        if b == 11:
            return False
        if b == 12:
            return False
        else:
            return False
    if a == 5:
        if b == 1:
            return True
        if b == 2:
            return True
        if b == 3:
            return False
        if b == 4:
            return False
        if b == 6:
            return True
        if b == 7:
            return False
        if b == 8:
            return False
        if b == 9:
            return True
        if b == 10:
            return True
        if b == 11:
            return False
        if b == 12:
            return False
        else:
            return False
    if a == 6:
        if b == 1:
            return True
        if b == 2:
            return True
        if b == 3:
            return True
        if b == 4:
            return False
        if b == 5:
            return True
        if b == 7:
            return True
        if b == 8:
            return False
        if b == 9:
            return True
        if b == 10:
            return True
        if b == 11:
            return True
        if b == 12:
            return False
        else:
            return False
    if a == 7:
        if b == 1:
            return False
        if b == 2:
            return True
        if b == 3:
            return True
        if b == 4:
            return True
        if b == 5:
            return False
        if b == 6:
            return True
        if b == 8:
            return True
        if b == 9:
            return False
        if b == 10:
            return True
        if b == 11:
            return True
        if b == 12:
            return True
        else:
            return False
    if a == 8:
        if b == 1:
            return False
        if b == 2:
            return False
        if b == 3:
            return True
        if b == 4:
            return True
        if b == 5:
            return False
        if b == 6:
            return False
        if b == 7:
            return True
        if b == 9:
            return False
        if b == 10:
            return False
        if b == 11:
            return True
        if b == 12:
            return True
        else:
            return False
    if a == 9:
        if b == 1:
            return False
        if b == 2:
            return False
        if b == 3:
            return False
        if b == 4:
            return False
        if b == 5:
            return True
        if b == 6:
            return True
        if b == 7:
            return False
        if b == 8:
            return False
        if b == 10:
            return True
        if b == 11:
            return False
        if b == 12:
            return False
        else:
            return False
    if a == 10:
        if b == 1:
            return False
        if b == 2:
            return False
        if b == 3:
            return False
        if b == 4:
            return False
        if b == 5:
            return True
        if b == 6:
            return True
        if b == 7:
            return True
        if b == 8:
            return False
        if b == 9:
            return True
        if b == 11:
            return True
        if b == 12:
            return False
        else:
            return False
    if a == 11:
        if b == 1:
            return False
        if b == 2:
            return False
        if b == 3:
            return False
        if b == 4:
            return False
        if b == 5:
            return False
        if b == 6:
            return True
        if b == 7:
            return True
        if b == 8:
            return True
        if b == 9:
            return False
        if b == 10:
            return True
        if b == 12:
            return True
        else:
            return False
    if a == 12:
        if b == 1:
            return False
        if b == 2:
            return False
        if b == 3:
            return False
        if b == 4:
            return False
        if b == 5:
            return False
        if b == 6:
            return False
        if b == 7:
            return True
        if b == 8:
            return True
        if b == 9:
            return False
        if b == 10:
            return False
        if b == 11:
            return True
        else:
            return False
    else:
            return False

if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    print args
    evaluate_mlp(args)
