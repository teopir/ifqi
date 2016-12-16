from __future__ import print_function
from ifqi.algorithms.pbo.gradpbo import GradPBO
import numpy as np
from scipy import optimize
import theano
import theano.tensor as T
from itertools import chain, combinations

# np.random.seed(14151)
from sklearn.preprocessing import PolynomialFeatures


def lqr_reg(s, a, theta):
    w1 = theta[0, 0]
    w2 = theta[0, 1]
    v = - w1 ** 2 * s * a - 0.5 * w2 * (a ** 2) - 0.4 * w2 * (s ** 2)
    return v.ravel()


def grad_lqr_reg(s, a, theta):
    w1 = theta[0]
    w2 = theta[1]
    g1 = -2 * w1 * s * a
    g2 = -0.5 * (a ** 2) - 0.4 * (s ** 2)
    return np.array([g1, g2])


def bellmanop(rho, theta):
    rho_s = np.reshape(rho, (theta.shape[1], -1))
    return np.dot(theta, rho_s)


def bellmanop_grad(rho, theta):
    return np.kron(theta.T, np.eye(rho.shape[0], rho.shape[1]))


def empirical_bop(s, a, r, snext, all_a, gamma, rho, theta):
    new_theta = bellmanop(rho, theta)
    qnop = lqr_reg(s, a, new_theta)
    bop = -np.ones(s.shape[0]) * np.inf
    for i in range(s.shape[0]):
        for j in range(a.shape[0]):
            qv = lqr_reg(snext[i], all_a[j], theta)
            if qv > bop[i]:
                bop[i] = qv
    v = qnop - r - gamma * bop
    return 0.5 * np.sum(v ** 2)


class LBPO(object):
    def __init__(self, init_rho):
        self.rho = theano.shared(value=np.array(init_rho, dtype=theano.config.floatX),
                                 borrow=True, name='rho')
        self.theta = T.matrix()
        self.outputs = [T.dot(self.theta, self.rho)]
        self.inputs = [self.theta]
        self.trainable_weights = [self.rho]

    def evaluate(self, theta):
        if not hasattr(self, "eval_f"):
            self.eval_f = theano.function(self.inputs, self.outputs[0])
        return self.eval_f(theta)


class LQRRegressor(object):
    def model(self, s, a, omega):
        q = - omega[:, 0] ** 2 * s * a - 0.5 * omega[:, 1] * a * a - 0.4 * omega[:, 1] * s * s
        return q.ravel()


class LQG_PBO(object):
    def __init__(self):
        self.theta = T.matrix()
        # define output for b
        combinations = PolynomialFeatures._combinations(2, 3, False, False)
        n_output_features_ = sum(1 for _ in combinations) + 1
        self.A_b = theano.shared(value=np.ones((n_output_features_,), dtype=theano.config.floatX),
                                 borrow=True, name='A_b')
        self.b_b = theano.shared(value=1.,
                                 borrow=True, name='b_b')

        combinations = PolynomialFeatures._combinations(2, 3, False, False)
        L = [(self.theta[:, 0] ** 0).reshape([-1, 1])]
        for i, c in enumerate(combinations):
            L.append(self.theta[:, c].prod(1).reshape([-1, 1]))
        self.XF3 = T.concatenate(L, axis=1)
        b = (T.dot(self.XF3, self.A_b) + self.b_b).reshape([-1, 1])

        # define output for k
        combinations = PolynomialFeatures._combinations(2, 2, False, False)
        n_output_features_ = sum(1 for _ in combinations) + 1
        self.rho_k = theano.shared(value=np.ones((n_output_features_,), dtype=theano.config.floatX),
                                   borrow=True, name='rho_k')

        combinations = PolynomialFeatures._combinations(2, 2, False, False)
        L = [(self.theta[:, 0] ** 0).reshape([-1, 1])]
        for i, c in enumerate(combinations):
            L.append(self.theta[:, c].prod(1).reshape([-1, 1]))
        self.XF2 = T.concatenate(L, axis=1)
        k = T.dot(self.XF2, self.rho_k).reshape([-1, 1])

        self.outputs = [T.concatenate([b, k], axis=1)]
        self.inputs = [self.theta]
        self.trainable_weights = [self.A_b, self.b_b, self.rho_k]

    def evaluate(self, theta):
        if not hasattr(self, "eval_f"):
            self.eval_f = theano.function(self.inputs, self.outputs[0])
        return self.eval_f(theta)


F = np.array([[1, 2], [3, 4]])
print(PolynomialFeatures(3).fit_transform(F))
lqgpbo = LQG_PBO()
print(lqgpbo.evaluate(F))

gamma = 0.99
rho = np.array([1., 2., 0., 3.]).reshape(2, 2)
theta = np.array([2., 0.2], dtype='float32').reshape(1, -1)

lbpo = LBPO(rho)  # bellman operator (apx)
q_model = LQRRegressor()  # q-function

s = np.array([1., 2., 3.]).reshape(-1, 1)
a = np.array([0., 3., 4.]).reshape(-1, 1)
nexts = s + 1
r = np.array([-1., -5., 0.])
actions = np.array([1, 2, 3]).reshape(-1, 1)  # discretization of the actions
# to be used for maximum estimate

# it is also possible to use keras model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim=2, init='uniform', activation='relu'))
model.add(Dense(2, init='uniform', activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# =================================================================
gpbo = GradPBO(bellman_model=lbpo, q_model=q_model, gamma=gamma)
assert np.allclose(bellmanop(rho, theta), gpbo.bopf(theta)), \
    '{}, {}'.format(bellmanop(rho, theta), gpbo.bopf(theta))
assert np.allclose(lqr_reg(s, a, theta), gpbo.qf(s, a, theta))

berr = gpbo.berrf(s, a, nexts, r, theta, actions)
tv = empirical_bop(s, a, r, nexts, actions, gamma, rho, theta)
assert np.allclose(berr, tv), '{}, {}'.format(berr, tv)

berr_grad = gpbo.grad_berrf(s, a, nexts, r, theta, actions)
eps = np.sqrt(np.finfo(float).eps)
f = lambda x: empirical_bop(s, a, r, nexts, actions, gamma, x, theta)
approx_grad = optimize.approx_fprime(rho.ravel(), f, eps).reshape(berr_grad[0].shape)
assert np.allclose(berr_grad, approx_grad), '{}, {}'.format(berr_grad, approx_grad)
