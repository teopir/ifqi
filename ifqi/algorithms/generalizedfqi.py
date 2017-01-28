from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import copy
import time

from keras.engine.training import slice_X, batch_shuffle, make_batches, \
    standardize_input_data, check_array_lengths
from keras import optimizers
from keras import callbacks as cbks

from .algorithm import Algorithm


class PBOHistory(cbks.Callback):
    def on_train_begin(self, logs={}):
        self.epoch = []
        self.batch = []
        self.hist = {}

    def on_batch_end(self, batch, logs={}):
        for k in self.params['metrics'] + ['theta']:
            if k in logs:
                self.hist.setdefault(k, []).append(logs[k])


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
                                         discrete_actions, gamma, horizon, verbose)
        # save MDP information
        self.norm_value = norm_value
        self.update_theta_every = update_theta_every if update_theta_every > 0 else -1

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
        self.discrete_actions = standardize_input_data(discrete_actions, ['discrete_actions'],
                                                       [(None,
                                                         self.action_dim)] if self.action_dim is not None else None,
                                                       exception_prefix='discrete_actions')

    def _make_train_function(self):
        """
        Construct the python train function from theano to be used in the fit process
        Returns:
            None
        """
        if self.train_function is None:
            print('compiling train function...')
            start = time.time()
            inputs = self._estimator.inputs + [self.T_Y]

            training_updates = self.optimizer.get_updates(self._estimator.trainable_weights,
                                                          {}, self.fqi_loss)

            # returns loss and metrics. Updates weights at each call.
            self.train_function = theano.function(inputs, [self.fqi_loss], updates=training_updates, name="trainer")
            print('compiled in {}s'.format(time.time() - start))

    def fit(self, sast, r,
            batch_size=32, nb_epoch=10, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True, theta_metrics={}):
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
        sast = standardize_input_data(sast, ['sast'], (None, 2 * self.state_dim + self.action_dim + 1),
                                      check_batch_dim=False, exception_prefix='sast')[0]

        next_states_idx = self.state_dim + self.action_dim
        sa = sast[:, :next_states_idx]
        s_next = sast[:, next_states_idx:-1]
        absorbing = sast[:, -1]

        # # prepare validation data
        # if validation_data:
        #     do_validation = True
        #     if len(validation_data) == 4:
        #         val_s, val_a, val_s_next, val_r = validation_data
        #     elif len(validation_data) == 5:
        #         val_s, val_a, val_s_next, val_r, val_theta = validation_data
        #     else:
        #         raise
        #
        #     val_s, val_a, val_s_next, val_r, val_theta = self._standardize_user_data(
        #         val_s, val_a, val_s_next, val_r, val_theta,
        #         check_batch_dim=False,
        #         batch_size=batch_size
        #     )
        #     self._make_test_function()
        #     val_f = self.test_function
        #     val_ins = val_s + val_a + val_s_next + [val_r]
        #
        # elif validation_split and 0. < validation_split < 1.:
        #     do_validation = True
        #     split_at = int(len(x[0]) * (1. - validation_split))
        #     x, val_x = (slice_X(x, 0, split_at), slice_X(x, split_at))
        #     y, val_y = (slice_X(y, 0, split_at), slice_X(y, split_at))
        #     sample_weights, val_sample_weights = (
        #         slice_X(sample_weights, 0, split_at), slice_X(sample_weights, split_at))
        #     self._make_test_function()
        #     val_f = self.test_function
        #     if self.uses_learning_phase and type(K.learning_phase()) is not int:
        #         val_ins = val_x + val_y + val_sample_weights + [0.]
        #     else:
        #         val_ins = val_x + val_y + val_sample_weights
        # else:
        #     do_validation = False
        #     val_f = None
        #     val_ins = None

        do_validation = False
        val_f = None
        val_ins = None

        ins = [sa]
        self._make_train_function()
        f = self.train_function

        # prepare display labels
        out_labels = ['loss']

        if do_validation:
            callback_metrics = copy.copy(out_labels) + ['val_' + n for n in out_labels]
        else:
            callback_metrics = copy.copy(out_labels)

        return self._fit_loop(f, ins, r, s_next, absorbing,
                              out_labels=out_labels,
                              batch_size=batch_size, nb_epoch=nb_epoch,
                              verbose=verbose, callbacks=callbacks,
                              val_f=val_f, val_ins=val_ins, shuffle=shuffle,
                              callback_metrics=callback_metrics,
                              theta_metrics=theta_metrics)

    def _fit_loop(self, f, ins, r, s_next, absorbing, out_labels=[],
                  batch_size=32, nb_epoch=100, verbose=1, callbacks=[],
                  val_f=None, val_ins=None, shuffle=True, callback_metrics=[],
                  theta_metrics={}):
        do_validation = False
        if val_f and val_ins:
            do_validation = True
            if verbose:
                print('Train on %d samples, validate on %d samples' %
                      (ins[0].shape[0], val_ins[0].shape[0]))

        nb_train_sample = ins[0].shape[0]
        print(nb_train_sample)
        index_array = np.arange(nb_train_sample)

        history = PBOHistory()
        callbacks = [history] + callbacks
        if verbose:
            callbacks += [cbks.ProgbarLogger()]
        callbacks = cbks.CallbackList(callbacks)
        callback_model = self

        callbacks._set_model(callback_model)
        callbacks._set_params({
            'batch_size': batch_size,
            'nb_epoch': nb_epoch,
            'nb_sample': nb_train_sample,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics + [el for el in theta_metrics.keys()]
        })
        callbacks.on_train_begin()
        callback_model.stop_training = False

        n_updates = 0

        maxq, maxa = self.maxQA(s_next, absorbing)

        if hasattr(self._estimator, 'adapt'):
            # update estimator structure
            self._estimator.adapt(iteration=self._iteration)

        # y = np.reshape(r + self.gamma * maxq, (-1, 1))
        y = r + self.gamma * maxq

        ins += [y]

        for epoch in range(nb_epoch):
            callbacks.on_epoch_begin(epoch)
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(nb_train_sample, batch_size)
            epoch_logs = {}
            for batch_index, (batch_start, batch_end) in enumerate(batches):
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

                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                for k in theta_metrics.keys():
                    batch_logs[k] = theta_metrics[k](self._estimator.get_weights())
                callbacks.on_batch_begin(batch_index, batch_logs)

                outs = f(*ins_batch)
                n_updates += 1

                if type(outs) != list:
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                if self.update_theta_every > 0 and n_updates % self.update_theta_every == 0:
                    maxq, maxa = self.maxQA(s_next, absorbing)

                    if hasattr(self._estimator, 'adapt'):
                        # update estimator structure
                        self._estimator.adapt(iteration=self._iteration)

                    # y = np.reshape(r + self.gamma * maxq, (-1, 1))
                    y = r + self.gamma * maxq
                    ins = [ins[0], y]

                callbacks.on_batch_end(batch_index, batch_logs)

                if batch_index == len(batches) - 1:  # last batch
                    # validation
                    if do_validation:
                        # replace with self._evaluate
                        val_outs = self._test_loop(val_f, val_ins,
                                                   batch_size=batch_size,
                                                   verbose=0)
                        if type(val_outs) != list:
                            val_outs = [val_outs]
                        # same labels assumed
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o

            callbacks.on_epoch_end(epoch, epoch_logs)

        callbacks.on_train_end()

        # finally apply the bellman operator K-times to get the final point
        if self._verbose > 1:
            print('learned theta: {}'.format(self._estimator.get_weights()))

        return history
