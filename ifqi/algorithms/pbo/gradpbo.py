from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np


class GradPBO(object):
    def __init__(self, bellman_model, q_model, gamma, optimizer):
        self.gamma = gamma
        s = T.dmatrix()
        a = T.dmatrix()
        s_next = T.dmatrix()
        r = T.dvector()
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
        self.s = s
        self.a = a
        self.s_next = s_next
        self.r = r
        self.all_actions = all_actions

    def _compute_max_q(self, s, all_actions, theta):
        q_values, _ = theano.scan(fn=lambda a, s, theta: self.q_model.model(s, a, theta),
                                  sequences=[all_actions], non_sequences=[s, theta])
        return T.max(q_values)
        # return q_values

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
            inputs = [self.s, self.a, self.s_next, self.r, self.theta, self.all_actions]

            training_updates = self.optimizer.get_updates(self.bellman_model.trainable_weights,
                                                          {}, self.berr)
            updates = training_updates

            # returns loss and metrics. Updates weights at each call.
            self.train_function = theano.function(inputs, [self.berr], updates=training_updates)

    def fit(self):
