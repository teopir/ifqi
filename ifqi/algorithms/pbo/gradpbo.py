from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np


class GradPBO(object):
    def linear_bellapx(self, rho, theta):
        return T.dot(rho, theta)

    def linear_qfunction(self, s, a, omega):
        q = - omega[0] ** 2 * s * a - 0.5 * omega[1] * a * a - 0.4 * omega[1] * s * s
        return q

    def bellman_error(self, s, a, nexts, r, rho, theta, gamma, all_actions):
        def compute_max_q(s, all_actions, theta):
            q_values, _ = theano.scan(fn=lambda s, a, theta: self.linear_qfunction(s, a, theta),
                                      sequences=[all_actions], non_sequences=[s, theta])
            return T.max(q_values)

        # compute new parameters
        c = self.linear_bellapx(rho, theta)
        # compute q-function with new parameters
        qbpo = self.linear_qfunction(s, a, c)
        # compute max over actions with old parameters
        # qmat = self.linear_qfunction(nexts, all_actions, theta)
        # qmax = T.max(qmat, axis=1)


        qmat, _ = theano.scan(fn=compute_max_q,
                              sequences=[nexts], non_sequences=[all_actions, theta])

        # compute empirical BOP
        v = qbpo - r - gamma * qmat
        # compute error
        err = 0.5 * T.sum(v ** 2)
        return err

    def __init__(self):
        self.actions = np.array([1, 2, 3])
        s = T.dmatrix()
        a = T.dmatrix()
        snext = T.dmatrix()
        r = T.dmatrix()
        rho = T.dmatrix()
        theta = T.dmatrix()
        all_actions = T.dmatrix()

        bop = self.linear_bellapx(rho, theta)
        q = self.linear_qfunction(s, a, theta)

        self.qf = theano.function([s, a, theta], q)
        self.bopf = theano.function([rho, theta], bop)

        self.berr = self.bellman_error(s, a, snext, r, rho, theta, 0.9, all_actions)
        self.berrf = theano.function([s, a, snext, r, rho, theta, all_actions], self.berr)
