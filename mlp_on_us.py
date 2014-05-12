
import random
import signal
import os
import h5py as h5

from breze.learn.mlp import Mlp
from breze.learn.trainer.trainer import Trainer
from breze.learn.trainer.report import KeyPrinter, JsonPrinter
import climin.initialize
import numpy as np
from sklearn.grid_search import ParameterSampler
from breze.learn.data import one_hot

def preamble(i):
    train_folder = os.path.dirname(os.path.realpath(__file__))
    module = os.path.join(train_folder, 'mlp_on_us.py')
    script = '/nthome/maugust/git/alchemie/scripts/alc.py'
    runner = 'python %s run %s' % (script, module)

    pre = '#SUBMIT: runner=%s\n' % runner
    pre += '#SUBMIT: gpu=no\n'

    minutes_before_3_hour = 15
    slurm_preamble = '#SBATCH -J MLP_2hiddens_on_us_%d\n' % (i)
    slurm_preamble += '#SBATCH --mem=15000\n'
    slurm_preamble += '#SBATCH --signal=INT@%d\n' % (minutes_before_3_hour*60)
    slurm_preamble += '#SBATCH --exclude=cn-7,cn-8\n'
    return pre + slurm_preamble



def draw_pars(n=1):
    class OptimizerDistribution(object):
        def rvs(self):
            grid = {
                'step_rate': [0.0001, 0.0005, 0.005,0.001,0.00001,0.00005],
                'momentum': [0.99, 0.995,0.9,0.95],
                'decay': [0.9, 0.95,0.99],
            }

            sample = list(ParameterSampler(grid, n_iter=1))[0]
            sample.update({'step_rate_max': 0.05, 'step_rate_min': 1e-7})
            return 'rmsprop', sample

    grid = {
        'n_hidden': [[200,200],[500,500],[1000,1000],[700,700],[100,100],[50,50]],
        'hidden_transfers': [['sigmoid','sigmoid'], ['tanh','tanh'], ['rectifier','rectifier']],
        'par_std': [1.5, 1, 1e-1, 1e-2,1e-3,1e-4,1e-5],
	'batch_size': [10000,5000,2000,1000],
        'optimizer': OptimizerDistribution(),
    }

    sampler = ParameterSampler(grid, n)
    return sampler


def load_data(pars):
   data = h5.File('../../usarray_data_scaled_train_val_bin.hdf5','r')
   X = data['trainig_set/train_set'][...][:330000]
   Z = data['trainig_labels/bin_train_labels'][...][:330000]
   VX = data['validation_set/val_set'][...][:140000]
   VZ = data['validation_labels/bin_val_labels'][...][:140000]
   Z = one_hot(Z,2)
   VZ = one_hot(VZ,2)


   return X, Z, VX, VZ


def generate_dict(trainer,data):
    trainer.val_key = 'val'
    trainer.eval_data = {}
    trainer.eval_data['train'] = ([data[0],data[1]])
    trainer.eval_data['val'] = ([data[2], data[3]])


def new_trainer(pars, data):
    X, Z, VX, VZ = data
    input_size = len(X[0])
    output_size = len(Z[0])
    batch_size = pars['batch_size']
    m = Mlp(input_size, pars['n_hidden'], output_size, 
            hidden_transfers=pars['hidden_transfers'], out_transfer='softmax',
            loss='cat_ce', batch_size = batch_size,
            optimizer=pars['optimizer'])
    climin.initialize.randomize_normal(m.parameters.data, 0, pars['par_std'])

    n_report = len(X)/batch_size
    max_iter = n_report * 1000

    interrupt = climin.stops.OnSignal()
    print dir(climin.stops)
    stop = climin.stops.Any([
        climin.stops.AfterNIterations(max_iter),
        climin.stops.OnSignal(signal.SIGTERM),
        #climin.stops.NotBetterThanAfter(1e-1,500,key='train_loss'),
    ])

    pause = climin.stops.ModuloNIterations(n_report)
    reporter = KeyPrinter(['n_iter', 'train_loss', 'val_loss'])

    t = Trainer(
        m,
        stop=stop, pause=pause, report=reporter,
        interrupt=interrupt)

    generate_dict(t,data)

    return t


def make_report(pars, trainer, data):
    return {'train_loss': trainer.score(trainer.eval_data['train']),
            'val_loss': trainer.score(trainer.eval_data['val'])}

