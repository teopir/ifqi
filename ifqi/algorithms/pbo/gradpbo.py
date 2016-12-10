from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np

class GradPBO(object):

    def linear_bellapx(self, rho, theta):
        return T.dot(rho, theta)

    def linear_qfunction(self, s, a, omega):
        q = - omega[0]**2 * s * a - 0.5 * omega[1] * a * a - 0.4 * omega[1] * s * s
        return q

    def bellman_error(self, s, a, nexts, r, rho, theta, gamma):
        def test(sprime, w):
            a_row = self.actions.reshape(1,-1)
            return self.linear_qfunction(sprime, a, w)

        c = self.linear_bellapx(rho, theta)
        qbpo = self.linear_qfunction(s, a, c)
        qmat = theano.scan(fn=test,
                           sequences=[nexts],non_sequences=theta)
        qmax = T.max(qmat, axis=1)
        v = qbpo - r - gamma * qmax
        err = 0.5 * T.sum(v**2)
        return err

    def __init__(self):
        self.actions = np.array([1,2,3])
        s = T.dmatrix()
        a = T.dmatrix()
        snext = T.dmatrix()
        r = T.dmatrix()
        rho = T.dmatrix()
        theta = T.dmatrix()

        bop = self.linear_bellapx(rho, theta)
        q = self.linear_qfunction(s, a, theta)

        self.qf = theano.function([s, a, theta], q)
        self.bopf = theano.function([rho, theta], bop)

        self.berr = self.bellman_error(s, a, snext, r, rho, theta, 0.9)





