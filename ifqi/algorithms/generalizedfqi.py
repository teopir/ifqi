from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import time
from six import iteritems

from keras.engine.training import slice_X, batch_shuffle, make_batches, \
    standardize_input_data, check_array_lengths
from keras import optimizers

from .algorithm import Algorithm


class GenGradFQI(Algorithm):
    """
    Construct a Generalized Gradient FQI instance given the specified parameters

    Args:
        q_model (object): A class representing the Q-function approximation
            It must comply with the following structure
            - inputs (list): attributes that defines the inputs (theano variables) of the model.
                             len(inputs) must be equal to 1.
            - outputs (list): attributes that defines the outputs (theano variables) of the model.
                             len(outputs) must be equal to 1.
            - trainable_weights (list): list of theano variables representing trainable weights.
        gamma (float): discount factor
        discrete_actions (numpy.array): discrete actions used to approximate the maximum (nactions, action_dim)
        optimizer: str (name of optimizer) or optimizer object.
                See [Keras optimizers](https://keras.io/optimizers/).
        state_dim (None, int): state dimension (state_dim)
        action_dim (None, int): action dimension (action_dim)
        norm_value (np.inf, int): defines the norm used to compute the error
        update_theta_every (int): is the number of steps of the gradient before to update theta.
                                  =1 it means that theta is updated at every gradient step (default 1)
        verbose (int): verbosity level
    """

    def __init__(self, estimator, gamma, discrete_actions,
                 optimizer="adam", state_dim=None, action_dim=None,
                 norm_value=2, update_theta_every=1, horizon=10,
                 verbose=0):
        super(GenGradFQI, self).__init__(estimator, state_dim, action_dim,
                                         discrete_actions, gamma, horizon,
                                         verbose)
        # save MDP information
        self.norm_value = norm_value
        self.update_theta_every = update_theta_every if update_theta_every > \
                                                        0 else -1

        # create theano variables
        self.T_Y = T.dvector()

        # define bellman operator (check that BOP has only one output)
        assert isinstance(estimator.inputs, list)
        assert len(estimator.inputs) == 1
        assert isinstance(estimator.outputs, list)
        assert len(estimator.outputs) == 1

        # construct (theano) Bellman error
        v = self._estimator.outputs[0] - self.T_Y
        if self.norm_value == np.inf:
            err = T.max(v ** 2)
        else:
            err = T.mean(v ** self.norm_value) ** (1. / self.norm_value)
        self.fqi_loss = err

        # define function to be used for train and drawing actions
        self.train_function = None

        # get keras optimizer
        self.optimizer = optimizers.get(optimizer)

        # validate input data (the output is a list storing the validated input)
        self.discrete_actions = standardize_input_data(
            discrete_actions,
            ['discrete_actions'],
            [(None, self.action_dim)] if self.action_dim is not None else None,
            exception_prefix='discrete_actions')

    def _make_train_function(self):
        """
        Construct the python train function from theano to be used
        in the fit process
        Returns:
            None
        """
        if self.train_function is None:
            print('compiling train function...')
            start = time.time()
            inputs = self._estimator.inputs + [self.T_Y]

            training_updates = self.optimizer.get_updates(
                self._estimator.trainable_weights,
                {}, self.fqi_loss)

            # returns loss and metrics. Updates weights at each call.
            self.train_function = theano.function(inputs, [self.fqi_loss],
                                                  updates=training_updates,
                                                  name="trainer")
            print('compiled in {}s'.format(time.time() - start))

    def fit(self, sast, r,
            batch_size=32, nb_epoch=10, shuffle=True,
            theta_metrics={}):
        """

        Args:
            s (numpy.array): the samples of the state (nsamples, state_dim)
            a (numpy.array): the samples of the state (nsamples, action_dim)
            s_next (numpy.array): the samples of the next (reached) state (nsamples, state_dim)
            r (numpy.array): the sample of the reward (nsamples, )
            theta (numpy.array): the sample of the Q-function parameters (1, n_params)
            batch_size (int): dimension of the batch used for a single step of the gradient
            nb_epoch (int): number of epochs
            verbose (int): 0 or 1. Verbosity mode. 0 = silent, 1 = verbose.
            callbacks (list): list of callbacks to be called during training.
                See [Keras Callbacks](https://keras.io/callbacks/).
            validation_split (float): float between 0 and 1:
                fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate the loss and any model metrics
                on this data at the end of each epoch.
            validation_data (tuple): data on which to evaluate the loss and any model metrics
                at the end of each epoch. The model will not be trained on this data.
                This could be a tuple (val_s, val_a, val_s_next, val_r) or a tuple
                (val_s, val_a, val_s_next, val_r, val_theta).
            shuffle (boolean): whether to shuffle the training data before each epoch.
            theta_metrics (dict): dictionary storing the pairs (name: callable object).
                The callable object/function is used to evaluate the Q-function parameters
                at each iteration. The signature of the callable is simple: f(theta)
                e.g.: theta_metrics={'k': lambda theta: evaluate(theta)})

        Returns:
            A PBOHistory instance storing train information
        """
        sast = standardize_input_data(sast, ['sast'], (
            None, 2 * self.state_dim + self.action_dim + 1),
                                      exception_prefix='sast')[0]

        next_states_idx = self.state_dim + self.action_dim
        sa = sast[:, :next_states_idx]
        s_next = sast[:, next_states_idx:-1]
        absorbing = sast[:, -1]

        n_updates = 0

        maxq, maxa = self.maxQA(s_next, absorbing)

        if hasattr(self._estimator, 'adapt'):
            # update estimator structure
            self._estimator.adapt(iteration=self._iteration)

        # y = np.reshape(r + self.gamma * maxq, (-1, 1))
        y = r + self.gamma * maxq

        ins = [sa, y]
        self._make_train_function()
        f = self.train_function

        nb_train_sample = sa.shape[0]
        index_array = np.arange(nb_train_sample)
        history = {"theta": []}
        for k in theta_metrics.keys():
            history.update({k: []})

        for epoch in range(nb_epoch):
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(nb_train_sample, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):

                if hasattr(self._estimator, '_model'):
                    ltheta = self._model.get_weights()
                    history["theta"].append(ltheta)
                else:
                    ltheta = self._estimator.get_weights()
                    history["theta"].append(ltheta)
                for k, v in iteritems(theta_metrics):
                    history[k].append(v(ltheta))

                batch_ids = index_array[batch_start:batch_end]
                try:
                    if type(ins[-1]) is float:
                        # do not slice the training phase flag
                        ins_batch = slice_X(ins[:-1], batch_ids) + [ins[-1]]
                    else:
                        ins_batch = slice_X(ins, batch_ids)
                except TypeError:
                    raise Exception('TypeError while preparing batch. '
                                    'If using HDF5 input data, '
                                    'pass shuffle="batch".')

                outs = f(*ins_batch)
                n_updates += 1

                if self.update_theta_every > 0 \
                        and n_updates % self.update_theta_every == 0:
                    maxq, maxa = self.maxQA(s_next, absorbing)

                    if hasattr(self._estimator, 'adapt'):
                        # update estimator structure
                        self._estimator.adapt(iteration=self._iteration)

                    # y = np.reshape(r + self.gamma * maxq, (-1, 1))
                    y = r + self.gamma * maxq
                    ins = [ins[0], y]

        if self._verbose > 1:
            print('learned theta: {}'.format(
                self._estimator.get_weights()))

        return history
