
import signal
import os
import h5py as h5
import numpy as np
import theano
import itertools
import theano.tensor as T

from breze.learn.mlp import Mlp
from breze.learn.trainer.trainer import Trainer
from breze.learn.trainer.report import KeyPrinter, JsonPrinter
import climin.initialize
from sklearn.grid_search import ParameterSampler
from breze.learn.data import one_hot

def preamble(i):
    train_folder = os.path.dirname(os.path.realpath(__file__))
    module = os.path.join(train_folder, 'mlp_2h_real_binning_svm.py')
    script = '/nthome/maugust/git/alchemie/scripts/alc.py'
    runner = 'python %s run %s' % (script, module)

    pre = '#SUBMIT: runner=%s\n' % runner
    pre += '#SUBMIT: gpu=no\n'

    minutes_before_3_hour = 15
    slurm_preamble = '#SBATCH -J MLP_2hiddens_on_us_real_%d\n' % (i)
    slurm_preamble += '#SBATCH --mem=10000\n'
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
        'L2': [0.1, 0.01, 0.001, .0001, .00001, .05, .005, .0005],
    }

    sampler = ParameterSampler(grid, n)
    return sampler


def load_data(pars):
   data = h5.File('/nthome/maugust/thesis/train_val_test_binning_real_int.hdf5','r')
   X = data['trainig_set/train_set']
   Z = data['trainig_labels/real_train_labels']
   VX = data['validation_set/val_set']
   VZ = data['validation_labels/real_val_labels']
   Z = one_hot(Z,13)
   VZ = one_hot(VZ,13)


   return (X, Z), (VX, VZ)


def make_data_dict(trainer,data):
    train_data, val_data = data
    trainer.val_key = 'val'
    trainer.eval_data = {}
    trainer.eval_data['train'] = ([data for data in train_data])
    trainer.eval_data['val'] = ([data for data in val_data])

def squared_hinge(target, prediction):
    return (T.maximum(1 - target * prediction, 0) ** 2)


class TangMlp(Mlp):

    def __init__(
        self, n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer, loss, optimizer, batch_size, noise_schedule,
        max_iter=1000, verbose=False):
        super(TangMlp, self).__init__(n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer, loss, False, optimizer, batch_size,
            max_iter, verbose)

        self.noise_schedule = noise_schedule

    def _make_args(self, X, Z, imp_weight=None):
        args = super(TangMlp, self)._make_args(X, Z)
        def corrupt(x, level):
            return x + np.random.normal(0, level, x.shape).astype(theano.config.floatX)
        return (((corrupt(x, n), z), k) for n, ((x, z), k) in itertools.izip(self.noise_schedule, args))


def new_trainer(pars, data):

    # 3700 for binning
    input_size = 3700
    # 13 as there are 12 fields
    output_size = 13
    batch_size = pars['batch_size']
    n_report = 40000/batch_size
    max_iter = n_report * 100

    noise_schedule = (max(1 - float(i) / max_iter, 1e-4) for i in itertools.count())

    m = TangMlp(input_size, pars['n_hidden'], output_size,
            hidden_transfers=pars['hidden_transfers'], out_transfer='identity',
            loss=squared_hinge, noise_schedule=noise_schedule, batch_size = batch_size,
            optimizer=pars['optimizer'])
    climin.initialize.randomize_normal(m.parameters.data, 0, pars['par_std'])

    weight_decay = ((m.parameters.in_to_hidden**2).sum()
                    + (m.parameters.hidden_to_hidden_0**2).sum()
                    + (m.parameters.hidden_to_out**2).sum())
    weight_decay /= m.exprs['inpt'].shape[0]
    m.exprs['true_loss'] = m.exprs['loss']
    c_wd = pars['L2']
    m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay

    # length of dataset should be 270000 (for no time-integration)



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

    make_data_dict(t,data)

    return t


def make_report(pars, trainer, data):
    return {'train_loss': trainer.score(*trainer.eval_data['train']),
            'val_loss': trainer.score(*trainer.eval_data['val'])}

