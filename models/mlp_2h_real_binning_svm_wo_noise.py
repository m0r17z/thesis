
import signal
import os
import h5py as h5

from breze.learn.mlp import Mlp
from breze.learn.trainer.trainer import Trainer
from breze.learn.trainer.report import KeyPrinter, JsonPrinter
import climin.initialize
from sklearn.grid_search import ParameterSampler
from breze.learn.data import one_hot
import theano.tensor as T
import numpy as np

def preamble(i):
    train_folder = os.path.dirname(os.path.realpath(__file__))
    module = os.path.join(train_folder, 'mlp_2h_real_binning_svm_wo_noise.py')
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

   Z = (Z * 2) - 1
   VZ = (VZ * 2) - 1

   return (X, Z), (VX, VZ)


def make_data_dict(trainer,data):
    train_data, val_data = data
    trainer.val_key = 'val'
    trainer.eval_data = {}
    trainer.eval_data['train'] = ([data for data in train_data])
    trainer.eval_data['val'] = ([data for data in val_data])


def new_trainer(pars, data):

    # 3700 for binning
    input_size = 3700
    # 13 as there are 12 fields
    output_size = 13
    batch_size = pars['batch_size']
    m = Mlp(input_size, pars['n_hidden'], output_size, 
            hidden_transfers=pars['hidden_transfers'], out_transfer='identity',
            loss='squared_hinge', batch_size = batch_size,
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
    n_report = 40000/batch_size
    max_iter = n_report * 100

    interrupt = climin.stops.OnSignal()
    print dir(climin.stops)
    stop = climin.stops.Any([
        climin.stops.AfterNIterations(max_iter),
        climin.stops.OnSignal(signal.SIGTERM),
        #climin.stops.NotBetterThanAfter(1e-1,500,key='train_loss'),
    ])

    pause = climin.stops.ModuloNIterations(n_report)
    reporter = KeyPrinter(['n_iter', 'train_loss', 'val_loss', 'emp_val_loss'])

    t = Trainer(
        m,
        stop=stop, pause=pause, report=reporter,
        interrupt=interrupt)

    make_data_dict(t,data)

    return t


def make_report(pars, trainer, data):
    data = h5.File('/nthome/maugust/thesis/train_val_test_binning_real_int.hdf5','r')
    TX = data['test_set/test_set']
    TZ = data['test_labels/real_test_labels']
    TZ = one_hot(TZ,13)
    current_pars = trainer.model.parameters.data
    trainer.model.parameters.data[...] = trainer.best_pars

    n_wrong = 1 - T.eq(T.argmax(trainer.model.exprs['output'], axis=1),
                               T.argmax(trainer.model.exprs['target'], axis=1)).mean()
    f_n_wrong = trainer.model.function(['inpt', 'target'], n_wrong)

    f_pos = T.mean(T.neq(T.argmax(trainer.model.exprs['output'], axis=1),0) * T.eq(T.argmax(trainer.model.exprs['target'], axis=1), 0))
    f_f_pos = trainer.model.function(['inpt', 'target'], f_pos)

    f_neg = T.mean(T.eq(T.argmax(trainer.model.exprs['output'], axis=1),0) * T.neq(T.argmax(trainer.model.exprs['target'], axis=1), 0))
    f_f_neg = trainer.model.function(['inpt', 'target'], f_neg)


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

    if relevant_fails > 0:
        neighbour_fails /= relevant_fails


    emp_loss_s = 'model achieved %f%% classification error on the test set' %emp_loss
    f_p_s = '\nmodel achieved %f%% false positives on the test set' %f_p
    f_n_s = '\nmodel achieved %f%% false negatives on the test set' %f_n
    neigh_s = '\nmodel achieved %f%% neighbour misspredictions on the test set' %neighbour_fails

    print emp_loss_s
    print f_p_s
    print f_n_s
    print neigh_s
    with open(os.path.join('.','eval_result.txt'),'w') as f:
        f.write(emp_loss_s)
        f.write(f_p_s)
        f.write(f_n_s)
        f.write(neigh_s)
    trainer.model.parameters.data[...] = current_pars

    return {'train_loss': trainer.score(*trainer.eval_data['train']),
            'val_loss': trainer.score(*trainer.eval_data['val']),
            'emp_test_loss': emp_loss}

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
