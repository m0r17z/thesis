# -*- coding: utf-8 -*-

"""Usage:
    mlp_eval.py <location>
"""

import docopt
import os
import gzip
import cPickle
import theano.tensor as T

from alchemie import contrib


def evaluate_mlp(args):
    dir = os.path.abspath(args['<location>'])
    os.chdir(dir)
    cps = contrib.find_checkpoints('.')
    if cps:
        with gzip.open(cps[-1], 'rb') as fp:
                trainer = cPickle.load(fp)
                trainer.model.parameters.data[...] = trainer.best_pars

                n_wrong = 1 - T.eq(T.argmax(trainer.model.exprs['output'], axis=1),
                                   T.argmax(trainer.model.exprs['target'], axis=1)).mean()
                f_n_wrong = trainer.model.function(['inpt', 'target'], n_wrong)

                result = f_n_wrong(*trainer.eval_data['val'])
                result_s = 'model achieved %f%% classification error on the validation set' %(result)
                print result_s
                with open(os.path.join(dir,'eval_mlp_result.txt'),'w') as f:
                    f.write(result_s)
    return 0


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    print args
    evaluate_mlp(args)