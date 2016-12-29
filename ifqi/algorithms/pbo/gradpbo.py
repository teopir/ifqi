from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np

from keras.engine.training import slice_X, batch_shuffle, make_batches, \
    standardize_input_data, check_array_lengths
from keras import optimizers


class GradPBO(object):
    def __init__(self, bellman_model, q_model, gamma,
                 discrete_actions,
                 optimizer,
                 state_dim=None, action_dim=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        s = T.dmatrix()
        a = T.dmatrix()
        s_next = T.dmatrix()
        r = T.dvector()
        # r = T.dmatrix()
        all_actions = T.dmatrix()
        self.bellman_model = bellman_model
        self.q_model = q_model

        # define bellman operator
        assert isinstance(bellman_model.inputs, list)
        assert len(bellman_model.inputs) == 1
        theta = bellman_model.inputs[0]

        self.berr = self.bellman_error(s, a, s_next, r, theta, self.gamma, all_actions)
        params = self.bellman_model.trainable_weights
        self.grad_berr = T.grad(self.berr, params)

        # compile all functions
        self.bopf = theano.function([theta], bellman_model.outputs[0])
        q = self.q_model.model(s, a, theta)
        self.qf = theano.function([s, a, theta], q)
        self.berrf = theano.function([s, a, s_next, r, theta, all_actions], self.berr)
        self.grad_berrf = theano.function([s, a, s_next, r, theta, all_actions], self.grad_berr)

        self.train_function = None
        self.draw_action_function = None
        self.s = s
        self.a = a
        self.s_next = s_next
        self.r = r
        self.all_actions = all_actions
        self.optimizer = optimizers.get(optimizer)
        self.all_actions_value = standardize_input_data(discrete_actions, ['all_actions'],
                                                        [(None,
                                                          self.action_dim)] if self.action_dim is not None else None,
                                                        check_batch_dim=False, exception_prefix='discrete_actions')

    def _compute_max_q(self, s, all_actions, theta):
        q_values, _ = theano.scan(fn=lambda a, s, theta: self.q_model.model(s, a, theta),
                                  sequences=[all_actions], non_sequences=[s, theta])
        return T.max(q_values)
        # return q_values

    def _compute_argmax_q(self, s, all_actions, theta):
        q_values, _ = theano.scan(fn=lambda a, s, theta: self.q_model.model(s, a, theta),
                                  sequences=[all_actions], non_sequences=[s, theta])
        return T.argmax(q_values)

    def bellman_error(self, s, a, nexts, r, theta, gamma, all_actions):
        # compute new parameters
        # c = self.linear_bellapx(rho, theta)
        c = self.bellman_model.outputs[0]
        # compute q-function with new parameters
        qbpo = self.q_model.model(s, a, c)

        # compute max over actions with old parameters
        qmat, _ = theano.scan(fn=self._compute_max_q,
                              sequences=[nexts], non_sequences=[all_actions, theta])

        # compute empirical BOP
        v = qbpo - r - gamma * qmat
        # compute error
        err = 0.5 * T.sum(v ** 2)
        return err

    def _make_train_function(self):
        if self.train_function is None:
            theta = self.bellman_model.inputs[0]
            inputs = [self.s, self.a, self.s_next, self.r, theta, self.all_actions]

            training_updates = self.optimizer.get_updates(self.bellman_model.trainable_weights,
                                                          {}, self.berr)
            updates = training_updates

            # returns loss and metrics. Updates weights at each call.
            self.train_function = theano.function(inputs, [self.berr], updates=training_updates)

    def fit(self, s, a, s_next, r, theta,
            batch_size=32, nb_epoch=10, verbose=1, shuffle=True):
        self._make_train_function()
        f = self.train_function

        s = standardize_input_data(s, ['s'], [(None, self.state_dim)] if self.state_dim is not None else None,
                                   check_batch_dim=False, exception_prefix='state')
        a = standardize_input_data(a, ['a'], [(None, self.action_dim)] if self.action_dim is not None else None,
                                   check_batch_dim=False, exception_prefix='action')
        # r = standardize_input_data(r, ['r'], [(None, 1)],
        #                            check_batch_dim=False, exception_prefix='reward')
        s_next = standardize_input_data(s_next, ['s_next'],
                                        [(None, self.state_dim)] if self.state_dim is not None else None,
                                        check_batch_dim=False, exception_prefix='state_next')
        theta = standardize_input_data(theta, ['theta'], (None, self.bellman_model.n_inputs()),
                                       check_batch_dim=False, exception_prefix='theta')
        all_actions = standardize_input_data(self.all_actions_value, ['all_actions'],
                                             [(None, self.action_dim)] if self.action_dim is not None else None,
                                             check_batch_dim=False, exception_prefix='discrete_actions')
        check_array_lengths(s, a, s_next)
        ins = s + a + s_next + [r]
        return self._fit_loop(f, ins, theta, all_actions,
                              batch_size, nb_epoch, verbose, shuffle)

    def _fit_loop(self, f, ins, theta, discrete_actions,
                  batch_size=32, nb_epoch=100, verbose=1, shuffle=True):
        nb_train_sample = ins[0].shape[0]
        print(nb_train_sample)
        index_array = np.arange(nb_train_sample)
        print(theta)
        for epoch in range(nb_epoch):
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(nb_train_sample, batch_size)
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
                inp = ins_batch + theta + discrete_actions
                outs = f(*inp)
                theta = [self.bellman_model.predict(theta)]
                print(theta)
                print('k: {}'.format(self.q_model.get_k(theta[0])))
                print(outs)
                print()

        self.learned_theta_value = theta[0]
        return theta

    def apply_bop(self, theta):
        return self.bellman_model.predict(theta)

    def _make_draw_action_function(self):
        if self.draw_action_function is None:
            # compute max over actions with old parameters
            theta = self.bellman_model.inputs[0]
            inputs = [self.s, theta, self.all_actions]
            idx_max, _ = theano.scan(fn=self._compute_argmax_q,
                                     sequences=[self.s], non_sequences=[self.all_actions, theta])
            self.draw_action_function = theano.function(inputs, [self.all_actions[idx_max]])

    def draw_action(self, state, done, flag):
        self._make_draw_action_function()
        state = standardize_input_data(state, ['state'],
                                       [(None, self.state_dim)] if self.state_dim is not None else None,
                                       check_batch_dim=False, exception_prefix='draw_state')
        return self.draw_action_function(state, self.learned_theta_value, self.all_actions_value)
