from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np


class LQGReg2P(object):
    def __init__(self, init_theta=[1., 1.]):
        self.w = np.array(init_theta)

    def predict(self, sa):
        k, b = self.w
        # print(k,b)
        return - b * b * sa[:, 0] * sa[:, 1] - 0.5 * k * sa[:, 1] ** 2 - 0.4 * k * sa[:, 0] ** 2

    def get_weights(self):
        return self.w

    def get_k(self, omega):
        b = omega[:, 0]
        k = omega[:, 1]
        return - b * b / k

    def set_weights(self, w):
        self.w = np.array(w)

    def count_params(self):
        return self.w.size

    def name(self):
        return "LQGRegressor2P"


class LQGReg2P_PBO(object):
    def model(self, s, a, omega):
        b = omega[:, 0]
        k = omega[:, 1]
        q = - b * b * s * a - 0.5 * k * a * a - 0.4 * k * s * s
        return q.ravel()

    def n_params(self):
        return 2

    def get_k(self, omega):
        b = omega[:, 0]
        k = omega[:, 1]
        return - b * b / k

    def name(self):
        return "LQGRegressor2P"


class LQGReg2P_GradFQI(object):
    def __init__(self, init_theta):
        self.theta = theano.shared(value=np.array(init_theta, dtype=theano.config.floatX),
                                   borrow=True, name='theta')
        self.inputs = [T.dmatrix()]
        self.outputs = [self.model(self.inputs[0])]
        self.trainable_weights = [self.theta]

    def model(self, X):
        q = - (self.theta[0] ** 2) * X[:, 0] * X[:, 1] - 0.5 * (self.theta[1]) * X[:, 1] * X[:, 1] \
            - 0.4 * (self.theta[1]) * X[:, 0] * X[:, 0]
        return q

    def predict(self, X, **kwargs):
        if not hasattr(self, "eval_f"):
            self.eval_f = theano.function(self.inputs, self.outputs[0])
        return self.eval_f(X).ravel()

    def get_k(self, theta):
        if isinstance(theta, list):
            theta = theta[0].eval()
        b = theta[:, 0]
        k = theta[:, 1]
        return - b * b / k

    def get_weights(self):
        return self.theta.eval()

    def name(self):
        return "LQGRegressor2P"
