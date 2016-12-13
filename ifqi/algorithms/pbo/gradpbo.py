from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np


class GradPBO(object):

    def bellman_error(self, s, a, nexts, r, theta, gamma, all_actions):
        def compute_max_q(s, all_actions, theta):
            q_values, _ = theano.scan(fn=lambda a, s, theta: self.q_model.model(s, a, theta),
                                      sequences=[all_actions], non_sequences=[s, theta])
            return T.max(q_values)
            # return q_values

        # compute new parameters
        # c = self.linear_bellapx(rho, theta)
        c = self.bellman_model.outputs[0]
        # compute q-function with new parameters
        qbpo = self.q_model.model(s, a, c)

        # compute max over actions with old parameters
        ## qmat = self.linear_qfunction(nexts, all_actions, theta)
        ## qmax = T.max(qmat, axis=1)
        qmat, _ = theano.scan(fn=compute_max_q,
                              sequences=[nexts], non_sequences=[all_actions, theta])

        # compute empirical BOP
        v = qbpo - r - gamma * qmat
        # compute error
        err = 0.5 * T.sum(v ** 2)
        return err

    def __init__(self, bellman_model, q_model, gamma):
        self.gamma = gamma
        self.actions = np.array([1, 2, 3])
        s = T.matrix()
        a = T.matrix()
        snext = T.matrix()
        r = T.dvector()
        all_actions = T.matrix()
        self.bellman_model = bellman_model
        self.q_model = q_model

        #define bellman operator
        theta = bellman_model.inputs
        assert isinstance(theta, list)
        assert len(theta) == 1
        theta = theta[0]
        bop = bellman_model.outputs[0]
        self.bopf = theano.function([theta], bop)

        #bop = self.linear_bellapx(rho, theta)

        q = self.q_model.model(s, a, theta)
        self.qf = theano.function([s, a, theta], q)


        params = self.bellman_model._collected_trainable_weights

        self.berr = self.bellman_error(s, a, snext, r, theta, self.gamma, all_actions)
        self.grad_berr = T.grad(T.sum(self.berr), params)
        self.berrf = theano.function([s, a, snext, r, theta, all_actions], self.berr, on_unused_input='ignore')
        self.grad_berrf = theano.function([s, a, snext, r, theta, all_actions], self.grad_berr, on_unused_input='ignore')


