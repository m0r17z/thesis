
import signal
import os
import h5py as h5

from breze.learn.cnn import Cnn
from breze.learn.trainer.trainer import Trainer
from breze.learn.trainer.report import KeyPrinter, JsonPrinter
import climin.initialize
from sklearn.grid_search import ParameterSampler
from breze.learn.data import one_hot

def preamble(i):
    train_folder = os.path.dirname(os.path.realpath(__file__))
    module = os.path.join(train_folder, 'cnn_2c2h_real_binning_patience.py')
    script = '/nthome/maugust/git/alchemie/scripts/alc.py'
    runner = 'python %s run %s' % (script, module)

    pre = '#SUBMIT: runner=%s\n' % runner
    pre += '#SUBMIT: gpu=no\n'

    minutes_before_3_hour = 15
    slurm_preamble = '#SBATCH -J CNN_2c2h_binning_real_%d\n' % (i)
    slurm_preamble += '#SBATCH --mem=10000\n'
    slurm_preamble += '#SBATCH --signal=INT@%d\n' % (minutes_before_3_hour*60)
    slurm_preamble += '#SBATCH --exclude=cn-7,cn-8\n'
    return pre + slurm_preamble



def draw_pars(n=1):
    class OptimizerDistribution(object):
        def rvs(self):
            grid = {
                'step_rate': [0.0001, 0.0005, 0.005,0.001,0.00001,0.00005],
                'momentum': [0.8, 0.99, 0.995,0.9,0.95],
                'decay': [0.8,0.9, 0.95,0.99],
            }

            sample = list(ParameterSampler(grid, n_iter=1))[0]
            sample.update({'step_rate_max': 0.05, 'step_rate_min': 1e-7})
            return 'rmsprop', sample

    grid = {
        'n_hidden_full': [[200,200],[500,500],[1000,1000],[700,700],[100,100],[50,50]],
        'n_hidden_conv': [[16,32],[32,64],[64,128],[16,64],[32,128],[16,128]],
        'hidden_full_transfers': [['sigmoid','sigmoid'], ['tanh','tanh'], ['rectifier','rectifier']],
        'hidden_conv_transfers': [['sigmoid','sigmoid'], ['tanh','tanh'], ['rectifier','rectifier']],
        'filter_shapes': [[[5,5],[5,5]],[[6,6],[6,6]],[[6,6],[5,5]],[[5,5],[4,4]],[[7,7],[6,6]],[[8,8],[7,7]]],
        'pool_size': [(2,2),(4,4),(8,8)],
        'par_std': [1.5, 1, 1e-1, 1e-2,1e-3,1e-4,1e-5],
	    'batch_size': [10000,5000,2000,1000],
        'optimizer': OptimizerDistribution(),
        'L2': [0.1, 0.01, 0.001, .0001, .00001, .05, .005, .0005],
    }

    sampler = ParameterSampler(grid, n)
    return sampler


def load_data(pars):
   data = h5.File('/nthome/maugust/thesis/train_val_test_binning_real_cnn_int.hdf5','r')
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


def new_trainer(pars, data):

    # 3700 for binning
    input_size = 3700
    # 13 as there are 12 fields
    output_size = 13
    n_channels = 2
    bin_cm = 10
    max_x_cm = 440
    min_x_cm = 70
    max_y_cm = 250
    x_range = max_x_cm/bin_cm - min_x_cm/bin_cm
    y_range = max_y_cm*2/bin_cm
    im_width = y_range
    im_height = x_range
    batch_size = pars['batch_size']
    m = Cnn(input_size, pars['n_hidden_conv'], pars['n_hidden_full'], output_size,
            pars['hidden_conv_transfers'], pars['hidden_full_transfers'], 'softmax',
            loss='cat_ce',image_height=im_height,image_width=im_width,n_image_channel=n_channels,pool_size=pars['pool_size'],filter_shapes=pars['filter_shapes'],
            batch_size = batch_size, optimizer=pars['optimizer'])
    climin.initialize.randomize_normal(m.parameters.data, 0, pars['par_std'])

    weight_decay = ((m.parameters.hidden_conv_to_hidden_full**2).sum()
                    + (m.parameters.hidden_full_to_hidden_full_0**2).sum()
                    + (m.parameters.hidden_to_out**2).sum())
    weight_decay /= m.exprs['inpt'].shape[0]
    m.exprs['true_loss'] = m.exprs['loss']
    c_wd = pars['L2']
    m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay

    # length of dataset should be 270000 (for no time-integration)
    n_report = 270000/batch_size
    max_iter = n_report * 100

    interrupt = climin.stops.OnSignal()
    print dir(climin.stops)
    stop = climin.stops.Any([
        climin.stops.Patience(m.exprs['val_loss'], max_iter),
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

