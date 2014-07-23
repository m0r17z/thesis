# -*- coding: utf-8 -*-

"""Module for learning various types of multilayer perceptrons."""


import itertools

import climin
import climin.util
import climin.gd
from climin.project import max_length_columns

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams

from breze.arch.model.neural import mlp
from breze.arch.model.varprop import mlp as varprop_mlp
from breze.arch.component.varprop.common import supervised_loss as varprop_supervised_loss
from breze.arch.component.common import supervised_loss
from breze.learn.base import SupervisedBrezeWrapperBase
from breze.arch.util import ParameterSet, Model


# TODO Mlp docs are loss missing


class Mlp(Model, SupervisedBrezeWrapperBase):
    """Multilayer perceptron class.

    This implementation uses a stack of affine mappings with a subsequent
    non linearity each.

    Parameters
    ----------

    n_inpt : integer
        Dimensionality of a single input.

    n_hiddens : list of integers
        List of ``k`` integers, where ``k`` is thenumber of layers. Each gives
        the size of the corresponding layer.

    n_output : integer
        Dimensionality of a single output.

    hidden_transfers : list, each item either string or function
        Transfer functions for each of the layers. Can be either a string which
        is then used to look up a transfer function in
        ``breze.component.transfer`` or a function that given a Theano tensor
        returns a tensor of the same shape.

    out_transfer : string or function
        Either a string to look up a function in ``breze.component.transfer``
        or a function that given a Theano tensor returns a tensor of the same
        shape.

    optimizer : string, pair
        Argument is passed to ``climin.util.optimizer`` to construct an
        optimizer.

    batch_size : integer, None
        Number of examples per batch when calculting the loss
        and its derivatives. None means to use all samples every time.

    imp_weight : boolean
        Flag indicating whether importance weights are used.

    max_iter : int
        Maximum number of optimization iterations to perform. Only respected
        during``.fit()``, not ``.iter_fit()``.

    verbose : boolean
        Flag indicating whether to print out information during fitting.
    """

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, loss,
                 imp_weight=False,
                 optimizer='lbfgs',
                 batch_size=None,
                 max_iter=1000, verbose=False):
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer
        self.loss = loss

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.imp_weight = imp_weight

        self.max_iter = max_iter
        self.verbose = verbose

        self.f_predict = None

        super(Mlp, self).__init__()

    def _init_pars(self):
        spec = mlp.parameters(self.n_inpt, self.n_hiddens, self.n_output)
        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

    def _init_exprs(self):
        self.exprs = {
            'inpt': T.matrix('inpt'),
            'target': T.matrix('target')
        }

        if self.imp_weight:
            self.exprs['imp_weight'] = T.matrix('imp_weight')

        P = self.parameters

        n_layers = len(self.n_hiddens)
        hidden_to_hiddens = [getattr(P, 'hidden_to_hidden_%i' % i)
                             for i in range(n_layers - 1)]
        hidden_biases = [getattr(P, 'hidden_bias_%i' % i)
                         for i in range(n_layers)]

        self.exprs.update(mlp.exprs(
            self.exprs['inpt'],
            P.in_to_hidden, hidden_to_hiddens, P.hidden_to_out,
            hidden_biases, P.out_bias,
            self.hidden_transfers, self.out_transfer))

        imp_weight = False if not self.imp_weight else self.exprs['imp_weight']
        self.exprs.update(supervised_loss(
            self.exprs['target'], self.exprs['output'], self.loss,
            imp_weight=imp_weight))


def dropout_optimizer_conf(
        steprate_0=1, steprate_decay=0.998, momentum_0=0.5,
        momentum_eq=0.99, n_momentum_anneal_steps=500,
        n_repeats=500):
    """Return a dictionary suitable for climin.util.optimizer which
    specifies the standard optimizer for dropout mlps."""
    steprate = climin.gd.decaying(steprate_0, steprate_decay)
    momentum = climin.gd.linear_annealing(
        momentum_0, momentum_eq, n_momentum_anneal_steps)

    # Define another time for steprate calculcation.
    momentum2 = climin.gd.linear_annealing(
        momentum_0, momentum_eq, n_momentum_anneal_steps)
    steprate = ((1 - j) * i for i, j in itertools.izip(steprate, momentum2))

    steprate = climin.gd.repeater(steprate, n_repeats)
    momentum = climin.gd.repeater(momentum, n_repeats)

    return 'gd', {
        'steprate': steprate,
        'momentum': momentum,
    }


class DropoutMlp(Mlp):
    """Class representing an MLP that is trained with dropout [D]_.

    The gist of this method is that hidden units and input units are "zerod out"
    with a certain probability.

    References
    ----------
    .. [D] Hinton, Geoffrey E., et al.
           "Improving neural networks by preventing co-adaptation of feature
           detectors." arXiv preprint arXiv:1207.0580 (2012).


    Attributes
    ----------

    Same attributes as an ``Mlp`` object.

    p_dropout_inpt : float
        Probability that an input unit is ommitted during a pass.

    p_dropout_hidden : float
        Probability that an input unit is ommitted during a pass.

    max_length : float
        Maximum squared length of a weight vector into a unit. After each
        update, the weight vectors will projected to be shorter.
    """

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, loss,
                 p_dropout_inpt=.2, p_dropout_hiddens=.5,
                 max_length=None,
                 optimizer='rprop',
                 batch_size=None,
                 max_iter=1000, verbose=False):
        """Create a DropoutMlp object.


        Parameters
        ----------

        Same attributes as an ``Mlp`` object.

        p_dropout_inpt : float
            Probability that an input unit is ommitted during a pass.

        p_dropout_hiddens : list of floats
            List of which each item gives the probability that a hidden unit
            of that layer is omitted during a pass.

        """
        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hiddens = p_dropout_hiddens
        super(DropoutMlp, self).__init__(
            n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer, loss,
            optimizer=optimizer, batch_size=batch_size, max_iter=max_iter,
            verbose=verbose)


class FastDropoutNetwork(Model, SupervisedBrezeWrapperBase):
    """Class representing an MLP that is trained with fast dropout [FD]_.

    This method employs a smooth approximation of dropout training.


    References
    ----------
    .. [FD] Wang, Sida, and Christopher Manning.
            "Fast dropout training."
            Proceedings of the 30th International Conference on Machine
            Learning (ICML-13). 2013.


    Attributes
    ----------

    Same attributes as an ``Mlp`` object.

    p_dropout_inpt : float
        Probability that an input unit is ommitted during a pass.

    p_dropout_hiddens : list of floats
        Each item constitues the probability that a hidden unit of the
        corresponding layer is ommitted during a pass.

    inpt_var : float
        Assumed variance of the inputs. "quasi zero" per default.
    """

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, loss,
                 optimizer='lbfgs',
                 batch_size=None,
                 p_dropout_inpt=.2,
                 p_dropout_hiddens=.5,
                 max_length=None,
                 inpt_var=1e-8,
                 max_iter=1000, verbose=False):
        """Create a FastDropoutMlp object.


        Parameters
        ----------

        Same parameters as an ``Mlp`` object.

        p_dropout_inpt : float
            Probability that an input unit is ommitted during a pass.

        p_dropout_hidden : float
            Probability that an input unit is ommitted during a pass.

        max_length : float or None
            Maximum squared length of a weight vector into a unit. After each
            update, the weight vectors will projected to be shorter.
            If None, no projection is performed.
        """
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer
        self.loss = loss

        self.p_dropout_inpt = p_dropout_inpt
        if isinstance(p_dropout_hiddens, float):
            self.p_dropout_hiddens = [p_dropout_hiddens]
        else:
            self.p_dropout_hiddens = p_dropout_hiddens

        if not all(0 < i < 1 for i in [p_dropout_inpt] + self.p_dropout_hiddens):
            raise ValueError('dropout rates have to be in (0, 1)')

        self.max_length = max_length
        self.inpt_var = inpt_var

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose

        super(FastDropoutNetwork, self).__init__()

    def _init_pars(self):
        spec = varprop_mlp.parameters(
            self.n_inpt, self.n_hiddens, self.n_output, False)
        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

    def _init_exprs(self):
        self.exprs = {
            'inpt_mean': T.matrix('inpt_mean'),
            'target': T.matrix('target')}
        P = self.parameters

        hidden_to_hiddens = [getattr(P, 'hidden_to_hidden_%i' % i)
                             for i in range(len(self.n_hiddens) - 1)]
        hidden_biases = [getattr(P, 'hidden_bias_%i' % i)
                         for i in range(len(self.n_hiddens))]
        inpt_var = T.zeros_like(self.exprs['inpt_mean']) + self.inpt_var

        self.exprs.update(varprop_mlp.exprs(
            self.exprs['inpt_mean'], inpt_var, self.exprs['target'],
            P.in_to_hidden,
            hidden_to_hiddens,
            P.hidden_to_out,
            hidden_biases,
            [1 for _ in hidden_biases],
            P.out_bias,
            1,
            self.hidden_transfers, self.out_transfer,
            self.p_dropout_inpt, self.p_dropout_hiddens))

        self.exprs['inpt'] = self.exprs['inpt_mean']

        self.exprs.update(varprop_supervised_loss(
            self.exprs['target'], self.exprs['output'], self.loss))

    def iter_fit(self, X, Z, info_opt=None):
        """Iteratively fit the parameters of the model to the given data with
        the given error function.

        Each iteration of the learning algorithm is an iteration of the returned
        iterator. The model is in a valid state after each iteration, so that
        the optimization can be broken any time by the caller.

        This method does `not` respect the max_iter attribute.

        Parameters
        ----------

        X : array_like
            Input data. 2D array of the shape ``(n ,d)`` where ``n`` is the
            number of data samples and ``d`` is the dimensionality of a single
            data sample.
        Z : array_like
            Target data. 2D array of the shape ``(n, l)`` array where ``n`` is
            defined as in ``X``, but ``l`` is the dimensionality of a single
            output.
        """
        for info in super(FastDropoutNetwork, self).iter_fit(X, Z, info_opt=info_opt):
            yield info
            if self.max_length is not None:
                W = self.parameters['in_to_hidden']
                max_length_columns(W, self.max_length)

                n_layers = len(self.n_hiddens)
                for i in range(n_layers - 1):
                    W = self.parameters['hidden_to_hidden_%i' % i]
                    max_length_columns(W, self.max_length)
                W = self.parameters['hidden_to_out']
                max_length_columns(W, self.max_length)

#theano.config.compute_test_value = 'warn'
#theano.config.exception_verbosity = 'high'
class MoE(Model, SupervisedBrezeWrapperBase):
    """Multilayer perceptron class.

    This implementation uses a stack of affine mappings with a subsequent
    non linearity each.

    Parameters
    ----------

    n_inpt : integer
        Dimensionality of a single input.

    n_hiddens : list of integers
        List of ``k`` integers, where ``k`` is thenumber of layers. Each gives
        the size of the corresponding layer.

    n_output : integer
        Dimensionality of a single output.

    hidden_transfers : list, each item either string or function
        Transfer functions for each of the layers. Can be either a string which
        is then used to look up a transfer function in
        ``breze.component.transfer`` or a function that given a Theano tensor
        returns a tensor of the same shape.

    out_transfer : string or function
        Either a string to look up a function in ``breze.component.transfer``
        or a function that given a Theano tensor returns a tensor of the same
        shape.

    optimizer : string, pair
        Argument is passed to ``climin.util.optimizer`` to construct an
        optimizer.

    batch_size : integer, None
        Number of examples per batch when calculting the loss
        and its derivatives. None means to use all samples every time.

    imp_weight : boolean
        Flag indicating whether importance weights are used.

    max_iter : int
        Maximum number of optimization iterations to perform. Only respected
        during``.fit()``, not ``.iter_fit()``.

    verbose : boolean
        Flag indicating whether to print out information during fitting.
    """

    def __init__(self, n_experts, n_inpt, n_hiddens, n_expert_hiddens, n_output, n_expert_output,
                 hidden_transfers, experts_hidden_transfers, out_transfer, experts_out_transfer, expert_loss, loss='custom',
                 imp_weight=False,
                 optimizer='lbfgs',
                 batch_size=None,
                 max_iter=1000, verbose=False):
        self.n_experts = n_experts
        self.n_inpt = n_inpt
        self.n_expert_hiddens = n_expert_hiddens
        self.n_hiddens = [n_hidden + n_expert_hidden*n_experts for n_hidden, n_expert_hidden in zip(n_hiddens, n_expert_hiddens)]
        self.n_output = n_output + n_expert_output*n_experts
        self.n_expert_output = n_expert_output
        self.hidden_transfers = hidden_transfers
        self.experts_hidden_transfers = experts_hidden_transfers
        self.out_transfer = out_transfer
        self.experts_out_transfer = experts_out_transfer
        self.loss = loss
        self.expert_loss = expert_loss

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.imp_weight = imp_weight

        self.max_iter = max_iter
        self.verbose = verbose

        self.f_predict = None

        super(MoE, self).__init__()

    def _init_pars(self):
        spec = mlp.parameters(self.n_inpt, self.n_hiddens, self.n_output)
        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
                self.parameters.data.shape).astype(theano.config.floatX)

    def _init_exprs(self):
        self.exprs = {
            'inpt': T.matrix('inpt'),
            'target': T.matrix('target')
        }

        #self.exprs['inpt'].tag.test_value = np.random.rand(5, 784)
        #self.exprs['target'].tag.test_value = np.random.rand(5, 10)

        if self.imp_weight:
            self.exprs['imp_weight'] = T.matrix('imp_weight')

        P = self.parameters

        n_layers = len(self.n_hiddens)
        hidden_to_hiddens = [getattr(P, 'hidden_to_hidden_%i' % i)
                                 for i in range(n_layers - 1)]
        hidden_biases = [getattr(P, 'hidden_bias_%i' % i)
                             for i in range(n_layers)]

        for exp_ind in np.arange(self.n_experts):
            in_col_from = self.n_expert_hiddens[0]*exp_ind
            in_col_to = self.n_expert_hiddens[0]*(exp_ind+1)
            out_row_from = self.n_expert_hiddens[-1]*exp_ind
            out_row_to = self.n_expert_hiddens[-1]*(exp_ind+1)
            out_col_from = self.n_expert_output*exp_ind
            out_col_to = self.n_expert_output*(exp_ind+1)

            exp_in_to_hidden = P.in_to_hidden[:, in_col_from:in_col_to]
            exp_hidden_to_hiddens = [hidden_to_hidden[self.n_expert_hiddens[i]*exp_ind:self.n_expert_hiddens[i]*(exp_ind+1),
                                     self.n_expert_hiddens[i+1]*exp_ind:self.n_expert_hiddens[i+1]*(exp_ind+1)]
                                     for i, hidden_to_hidden in enumerate(hidden_to_hiddens)]
            exp_hidden_to_out = P.hidden_to_out[out_row_from:out_row_to, out_col_from:out_col_to]

            exp_hidden_biases = [hidden_bias[self.n_expert_hiddens[i]*exp_ind:self.n_expert_hiddens[i]*(exp_ind+1)]
                                     for i, hidden_bias in enumerate(hidden_biases)]
            exp_out_bias = P.out_bias[self.n_expert_output*exp_ind:self.n_expert_output*(exp_ind+1)]

            exp_dict = mlp.exprs(self.exprs['inpt'], exp_in_to_hidden, exp_hidden_to_hiddens, exp_hidden_to_out,
                exp_hidden_biases, exp_out_bias, self.experts_hidden_transfers, self.experts_out_transfer)

            for key in exp_dict.keys():
                exp_dict[key +'_exp_%d' %exp_ind] = exp_dict[key]
                exp_dict.pop(key)
            self.exprs.update(exp_dict)

            exp_loss_dict = supervised_loss(self.exprs['target'], self.exprs['output_exp_%d' %exp_ind], self.expert_loss)
            for key in exp_loss_dict.keys():
                exp_loss_dict[key +'_exp_%d' %exp_ind] = exp_loss_dict[key]
                exp_loss_dict.pop(key)
            self.exprs.update(exp_loss_dict)

        man_in_to_hidden = P.in_to_hidden[:, self.n_expert_hiddens[0]*self.n_experts:]
        man_hidden_to_hiddens = [hidden_to_hidden[self.n_expert_hiddens[i]*self.n_experts:,
                                     self.n_expert_hiddens[i+1]*self.n_experts:]
                                     for i, hidden_to_hidden in enumerate(hidden_to_hiddens)]
        man_hidden_to_out = P.hidden_to_out[self.n_expert_hiddens[-1]*self.n_experts:,
                            self.n_expert_output*self.n_experts:]

        man_hidden_biases = [hidden_bias[self.n_expert_hiddens[i]*self.n_experts:]
                                     for i, hidden_bias in enumerate(hidden_biases)]
        man_out_bias = P.out_bias[self.n_expert_output*self.n_experts:]

        man_dict = mlp.exprs(self.exprs['inpt'], man_in_to_hidden, man_hidden_to_hiddens, man_hidden_to_out,
                man_hidden_biases, man_out_bias, self.hidden_transfers, self.out_transfer)

        for key in man_dict.keys():
                man_dict[key +'_man'] = man_dict[key]
                man_dict.pop(key)
        self.exprs.update(man_dict)

        self.exprs['loss_sample_wise'] = T.concatenate([T.reshape(self.exprs['loss_sample_wise_exp_%d' %exp], (self.exprs['loss_sample_wise_exp_%d' %exp].shape[0],1))
                                                        for exp in np.arange(self.n_experts)], axis=1)
        self.exprs['loss_sample_wise'] = T.sum(self.exprs['loss_sample_wise']*self.exprs['output_man'], axis=1)
        self.exprs['loss'] = self.exprs['loss_sample_wise'].mean()

        chosen_experts = T.argmax(self.exprs['output_man'], axis=1)
        selection_matrix = T.zeros((chosen_experts.shape[0],3))
        selection_matrix = T.set_subtensor(selection_matrix[T.arange(selection_matrix.shape[0]), chosen_experts], 1)
        selection_matrix = T.reshape(selection_matrix, (selection_matrix.shape[0], selection_matrix.shape[1], 1))
        expert_outputs = T.concatenate([T.reshape(self.exprs['output_exp_%d' %exp], (self.exprs['output_exp_%d' %exp].shape[0],1,10)) for exp in np.arange(self.n_experts)], axis=1)
        chosen_outputs = selection_matrix * expert_outputs
        self.exprs['output'] = T.sum(chosen_outputs, axis=1)

