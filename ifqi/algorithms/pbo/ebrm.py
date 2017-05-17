from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import copy

from keras.engine.training import slice_X, batch_shuffle, make_batches, \
    standardize_input_data, check_array_lengths
from keras import optimizers
from keras import callbacks as cbks


class PBOHistory(cbks.Callback):
    def on_train_begin(self, logs={}):
        self.epoch = []
        self.batch = []
        self.hist = {}

    def on_batch_end(self, batch, logs={}):
        for k in self.params['metrics'] + ['theta']:
            if k in logs:
                self.hist.setdefault(k, []).append(logs[k])


class EmpiricalBellmanResidualMinimization(object):
    """EBRM
    """

    def __init__(self, q_model, gamma,
                 discrete_actions,
                 optimizer,
                 state_dim=None, action_dim=None, incremental=True):
        # save MDP information
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.incremental = incremental
        self.gamma = gamma

        # create theano variables
        T_s = T.matrix()
        T_a = T.matrix()
        T_s_next = T.matrix()
        T_r = T.vector()
        # T_r = T.dmatrix()
        T_discrete_actions = T.matrix()

        # store models of bellman apx and Q-function
        self.q_model = q_model

        # construct (theano) Bellman error
        self.T_bellman_err = self.bellman_error(T_s, T_a, T_s_next, T_r, self.gamma, T_discrete_actions)

        # define function to be used for train and drawing actions
        self.train_function = None
        self.draw_action_function = None

        self.T_s = T_s
        self.T_a = T_a
        self.T_s_next = T_s_next
        self.T_r = T_r
        self.T_discrete_actions = T_discrete_actions

        # get keras optimizer
        self.optimizer = optimizers.get(optimizer)

        # validate input data (the output is a list storing the validated input)
        self.discrete_actions = standardize_input_data(discrete_actions, ['discrete_actions'],
                                                       [(None,
                                                         self.action_dim)] if self.action_dim is not None else None,
                                                       check_batch_dim=False, exception_prefix='discrete_actions')

    def _compute_max_q(self, s, discrete_actions):
        """
        Compute the maximum value of the Q-function in the given state.

        Args:
            s (theano.matrix): the state matrix (1 x state_dim)
            discrete_actions (theano.matrix): the discrete actions (nactions x action_dim)

        Returns:
            The maximum values of the Q-function in the given state w.r.t. all the discrete actions

        """
        q_values, _ = theano.scan(fn=lambda a, s: self.q_model.model(s, a),
                                  sequences=[discrete_actions], non_sequences=[s])
        return T.max(q_values)
        # return q_values

    def _compute_argmax_q(self, s, discrete_actions):
        """
        Compute the index of the action with the maximum Q-value in the given state.

        Args:
            s (theano.matrix): the state matrix (1 x state_dim)
            discrete_actions (theano.matrix): the discrete actions (nactions x action_dim)

        Returns:
            The index of the action that maximixes the Q-function in the given state
        """
        q_values, _ = theano.scan(fn=lambda a, s: self.q_model.model(s, a),
                                  sequences=[discrete_actions], non_sequences=[s])
        return T.argmax(q_values)

    def bellman_error(self, s, a, nexts, r, gamma, discrete_actions):
        """
        Compute the symbolic expression of the Bellman error.

        Args:
            s (theano.matrix): the state matrix (nsamples x state_dim)
            a (theano.matrix): the action matrix (nsamples x state_dim)
            nexts (theano.matrix): the next state matrix (nsamples x state_dim)
            r (theano.vector): the reward vector (nsamples)
            gamma (float): discount factor
            discrete_actions (theano.matrix): the discrete actions (nactions x action_dim)

        Returns:
            The theano expression of the Bellman error
        """
        # compute q-function
        qbpo = self.q_model.model(s, a).ravel()

        # compute max over actions with old parameters
        qmat, _ = theano.scan(fn=self._compute_max_q,
                              sequences=[nexts], non_sequences=[discrete_actions])

        # compute empirical BOP
        v = qbpo - r - gamma * qmat
        # compute error
        err = 0.5 * T.mean(v ** 2)
        # err = T.max(T.abs_(v))
        return err

    def _make_train_function(self):
        """
        Construct the python train function from theano to be used in the fit process
        Returns:
            None
        """
        if self.train_function is None:
            inputs = [self.T_s, self.T_a, self.T_s_next, self.T_r, self.T_discrete_actions]

            training_updates = self.optimizer.get_updates(self.q_model.trainable_weights,
                                                          {}, self.T_bellman_err)

            # returns loss and metrics. Updates weights at each call.
            self.train_function = theano.function(inputs, [self.T_bellman_err], updates=training_updates)

    def _standardize_user_data(self, s, a, s_next, r, check_batch_dim=False):
        """

        Args:
            s (numpy.array): the samples of the state (nsamples, state_dim)
            a (numpy.array): the samples of the state (nsamples, action_dim)
            s_next (numpy.array): the samples of the next (reached) state (nsamples, state_dim)
            r (numpy.array): the sample of the reward (nsamples, )
            check_batch_dim (bool): default False

        Returns:
            The standardized values (s, a, s_next, r, theta)

        """
        s = standardize_input_data(s, ['s'], [(None, self.state_dim)] if self.state_dim is not None else None,
                                   check_batch_dim=check_batch_dim, exception_prefix='state')
        a = standardize_input_data(a, ['a'], [(None, self.action_dim)] if self.action_dim is not None else None,
                                   check_batch_dim=check_batch_dim, exception_prefix='action')
        # r = standardize_input_data(r, ['r'], [(None, 1)],
        #                            check_batch_dim=False, exception_prefix='reward')
        s_next = standardize_input_data(s_next, ['s_next'],
                                        [(None, self.state_dim)] if self.state_dim is not None else None,
                                        check_batch_dim=check_batch_dim, exception_prefix='state_next')
        check_array_lengths(s, a, s_next)
        return s, a, s_next, r

    def fit(self, s, a, s_next, r,
            batch_size=32, nb_epoch=10, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True, theta_metrics={}):
        """

        Args:
            s (numpy.array): the samples of the state (nsamples, state_dim)
            a (numpy.array): the samples of the state (nsamples, action_dim)
            s_next (numpy.array): the samples of the next (reached) state (nsamples, state_dim)
            r (numpy.array): the sample of the reward (nsamples, )
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
        s, a, s_next, r = self._standardize_user_data(
            s, a, s_next, r,
            check_batch_dim=False
        )

        all_actions = standardize_input_data(self.discrete_actions, ['all_actions'],
                                             [(None, self.action_dim)] if self.action_dim is not None else None,
                                             check_batch_dim=False, exception_prefix='discrete_actions')

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

        ins = s + a + s_next + [r]
        self._make_train_function()
        f = self.train_function

        # prepare display labels
        out_labels = ['bellman_error']

        if do_validation:
            callback_metrics = copy.copy(out_labels) + ['val_' + n for n in out_labels]
        else:
            callback_metrics = copy.copy(out_labels)

        return self._fit_loop(f, ins, all_actions,
                              out_labels=out_labels,
                              batch_size=batch_size, nb_epoch=nb_epoch,
                              verbose=verbose, callbacks=callbacks,
                              val_f=val_f, val_ins=val_ins, shuffle=shuffle,
                              callback_metrics=callback_metrics,
                              theta_metrics=theta_metrics)

    def _fit_loop(self, f, ins, discrete_actions, out_labels=[],
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
            'metrics': callback_metrics + [el for el in theta_metrics.keys()],
            'theta': self.q_model.trainable_weights[0].eval()
        })
        callbacks.on_train_begin()
        callback_model.stop_training = False

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
                batch_logs['theta'] = self.q_model.trainable_weights[0].eval()
                for k in theta_metrics.keys():
                    batch_logs[k] = theta_metrics[k](self.q_model.trainable_weights)
                callbacks.on_batch_begin(batch_index, batch_logs)

                inp = ins_batch + discrete_actions
                outs = f(*inp)

                if type(outs) != list:
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

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
        return history

    def _make_draw_action_function(self):
        """
        Construct the python function from theano to be used in the draw_action process
        Returns:
            None
        """
        if self.draw_action_function is None:
            # compute max over actions with old parameters
            idx_max, _ = theano.scan(fn=self._compute_argmax_q,
                                     sequences=[self.T_s], non_sequences=[self.T_discrete_actions])
            inputs = [self.T_s, self.T_discrete_actions]
            self.draw_action_function = theano.function(inputs, [self.T_discrete_actions[idx_max]])

    def draw_action(self, state, done, flag):
        """
        Samples the action to be executed.
        Args:
            state (numpy.array): the state to be evaluated (1, state_dim) or (state_dim,)
            done: ??
            flag: ??

        Returns:
            The action to be executed in the state
        """
        state = state.astype(theano.config.floatX)
        self._make_draw_action_function()
        state = standardize_input_data(state, ['state'],
                                       [(None, self.state_dim)] if self.state_dim is not None else None,
                                       check_batch_dim=False, exception_prefix='draw_state')
        return self.draw_action_function(state[0], self.discrete_actions[
            0])  # we take index zero since they are lists of numpy matrices

    def _make_additional_functions(self):
        # compile all functions
        self.F_q = theano.function([self.T_s, self.T_a], self.q_model.model(self.T_s, self.T_a))
        self.F_bellman_err = theano.function(
            [self.T_s, self.T_a, self.T_s_next, self.T_r, self.T_discrete_actions], self.T_bellman_err, on_unused_input='ignore')
        T_grad_bellman_err = T.grad(self.T_bellman_err, self.q_model.trainable_weights)
        self.F_grad_bellman_berr = theano.function(
            [self.T_s, self.T_a, self.T_s_next, self.T_r, self.T_discrete_actions], T_grad_bellman_err)
