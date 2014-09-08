# -*- coding: utf-8 -*-

"""Usage:
    find_best_deep_svm.py <location>
"""


import cPickle
import gzip
import os
import sys

import docopt
import numpy as np



def evaluate(args):
    dir = os.path.abspath(args['<location>'])
    sub_dirs = [os.path.join(dir, sub_dir)
                       for sub_dir in os.listdir(dir)]
    best_loss = np.inf
    best_exp = ''

    for sub_dir in sub_dirs:
        if not os.path.isdir(sub_dir):
            continue
        os.chdir(sub_dir)
        with open('./eval_result.txt') as f:
            test_loss = float(f.readline().replace('model achieved ', '').replace('% classification error on the test set',''))

            if test_loss < best_loss:
                best_loss = test_loss
                best_exp = sub_dir
                
    r_string = '>>> found the best deep svm in\n>>> %s\n>>> with a test loss of %f' %(best_exp, best_loss)
    print r_string
    with open(os.path.join(dir, 'result.txt'),'w') as result:
        result.write(r_string)

def main(args):

    evaluate(args)
    exit_code = 0

    return exit_code


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    print args
    sys.exit(main(args))