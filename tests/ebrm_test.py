from __future__ import print_function
from ifqi.algorithms.pbo.ebrm import EmpiricalBellmanResidualMinimization
import numpy as np
import theano
import theano.tensor as T
from time import time
from scipy import optimize
from keras import activations

theano.config.floatX = 'float64'


def lqr_reg(s, a, theta):
    theta = theta[0]
    theta = theta.reshape(-1, 2)
    w1 = theta[0, 0]
    w2 = theta[0, 1]
    v = - w1 ** 2 * s * a - 0.5 * w2 * (a ** 2) - 0.4 * w2 * (s ** 2)
    return v.ravel()


def NN_output(s, a, L):
    if len(L) == 2:
        A, B = L
    else:
        L = L[0]
        A = np.array(L[0:2]).reshape(2,1)
        B = np.array(L[2]).reshape(1,1)
    x = np.column_stack((s, a))
    return np.dot(x, A) + B


def grad_lqr_reg(s, a, theta):
    w1 = theta[0]
    w2 = theta[1]
    g1 = -2 * w1 * s * a
    g2 = -0.5 * (a ** 2) - 0.4 * (s ** 2)
    return np.array([g1, g2])


def empirical_bop(s, a, r, snext, all_a, gamma, funct, theta):
    qnop = funct(s, a, theta)
    bop = -np.ones(s.shape[0]) * np.inf
    for i in range(s.shape[0]):
        for j in range(a.shape[0]):
            qv = funct(snext[i], all_a[j], theta)
            if qv > bop[i]:
                bop[i] = qv
    v = qnop.ravel() - r - gamma * bop.ravel()
    # print('qnop', qnop.ravel())
    # print('bop', bop.ravel())
    # print('v', v.ravel())
    be = 0.5 * np.mean(v ** 2)
    return be


class LQG_NN(object):
    def __init__(self, input_dim, output_dim,
                 layers=[20], activations=['relu']):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = layers
        self.activations = activations
        self.trainable_weights = self.init()

    def init(self):
        # define the shared parameters
        self.W, self.b = [], []
        params = []
        for i in range(len(self.layers)):
            self.W.append(self.init_weights((self.input_dim if i == 0 else self.layers[i - 1], self.layers[i]),
                                            sigma=0, name='W_{}'.format(i)))
            params.append(self.W[-1])
            self.b.append(theano.shared(value=np.zeros((self.layers[i],), dtype=theano.config.floatX),
                                        borrow=True, name='b_{}'.format(i)))
            params.append(self.b[-1])
        last_layer_dim = self.input_dim
        if len(self.layers) > 0:
            last_layer_dim = self.layers[-1]
        self.Wy = self.init_weights((last_layer_dim, self.output_dim), sigma=0, name='Wy')
        self.by = theano.shared(value=np.zeros((self.output_dim,), dtype=theano.config.floatX),
                                borrow=True, name='by')
        # params = self.W + self.b + [self.Wy, self.by]
        params += [self.Wy, self.by]
        return params

    def model(self, s, a):
        s = s.reshape((-1, 1), ndim=2)
        a = a.reshape((-1, 1), ndim=2)
        y = T.concatenate((s, a), axis=1)
        for i in range(len(self.layers)):
            act = activations.get(self.activations[i])
            y = act(T.dot(y, self.W[i]) + self.b[i])
        act = activations.get("linear")
        y = act(T.dot(y, self.Wy) + self.by)
        return y

    def floatX(self, arr):
        return np.asarray(arr, dtype=theano.config.floatX)

    def init_weights(self, shape, sigma=0.01, name=''):
        if sigma == 0:
            W_bound = np.sqrt(6. / (shape[0] + shape[1]))
            return theano.shared(self.floatX(np.random.uniform(low=-W_bound, high=W_bound, size=shape)),
                                 borrow=True, name=name)
        return theano.shared(self.floatX(np.random.randn(*shape) * sigma), borrow=True, name=name)

    def get_k(self, theta):
        return 1.

    def evaluate(self, s, a):
        if not hasattr(self, "eval_f"):
            T_s = T.fmatrix()
            T_a = T.fmatrix()
            self.eval_f = theano.function([T_s, T_a], self.model(T_s, T_a))
        return self.eval_f(s, a)


class LQRRegressor(object):
    def __init__(self, init_theta):
        self.theta = theano.shared(value=np.array(init_theta, dtype=theano.config.floatX),
                                   borrow=True, name='theta')
        self.trainable_weights = [self.theta]

    def model(self, s, a):
        q = - self.theta[:, 0] ** 2 * s * a - 0.5 * self.theta[:, 1] * a * a - 0.4 * self.theta[:, 1] * s * s
        return q.ravel()

    def evaluate(self, s, a):
        if not hasattr(self, "eval_f"):
            T_s = T.fmatrix()
            T_a = T.fmatrix()
            self.eval_f = theano.function([T_s, T_a], self.model(T_s, T_a))
        return self.eval_f(s, a)


def check_v(v1, v2, verbose=0):
    if verbose > 0:
        print(v1)
        print(v2)
    assert np.allclose(v1, v2), '{} != {}'.format(v1, v2)


if __name__ == "__main__":
    gamma = 0.99
    theta = np.array([2., 0.2], dtype='float32').reshape(1, -1)

    s = np.array([1., 2., 3.], dtype=theano.config.floatX).reshape(-1, 1)
    a = np.array([0., 3., 4.], dtype=theano.config.floatX).reshape(-1, 1)
    nexts = (s + 1).copy()
    r = np.array([-1., -5., 0.], dtype=theano.config.floatX)
    discrete_actions = np.array([1., 2., 3.9], dtype=theano.config.floatX).reshape(-1,
                                                                                   1)  # discretization of the actions
    # to be used for maximum estimate
    # print(s,a,nexts,r,discrete_actions)

    q_model = LQRRegressor(theta)  # q-function

    pfpo = EmpiricalBellmanResidualMinimization(q_model=q_model,
                                                discrete_actions=discrete_actions,
                                                gamma=gamma, optimizer="adam",
                                                state_dim=1, action_dim=1)
    start = time()
    pfpo._make_additional_functions()
    print('compilation time: {}'.format(time() - start))

    check_v(pfpo.F_q(s, a), lqr_reg(s, a, [q_model.theta.eval()]))

    print('\n--- checking bellman error')
    berr = pfpo.F_bellman_err(s, a, nexts, r, discrete_actions)
    tv = empirical_bop(s, a, r, nexts, discrete_actions, gamma, lqr_reg, [q_model.theta.eval()])
    check_v(berr, tv, 1)

    print('\n--- checking gradient of the bellman error')
    berr_grad = pfpo.F_grad_bellman_berr(s, a, nexts, r, discrete_actions)
    eps = np.sqrt(np.finfo(float).eps)
    f = lambda x: empirical_bop(s, a, r, nexts, discrete_actions, gamma, lqr_reg, [x])
    approx_grad = optimize.approx_fprime(q_model.theta.eval().ravel(), f, eps).reshape(berr_grad[0].shape)
    check_v(berr_grad, approx_grad, 1)

    print()
    print('--' * 30)
    print('CHECKING NN MODEL')
    print('--' * 30)

    q_model = LQG_NN(2, 1, layers=[], activations=[])  # q-function

    pfpo = EmpiricalBellmanResidualMinimization(q_model=q_model,
                                                discrete_actions=discrete_actions,
                                                gamma=gamma, optimizer="adam",
                                                state_dim=1, action_dim=1)
    start = time()
    pfpo._make_additional_functions()
    print('compilation time: {}'.format(time() - start))

    W = q_model.trainable_weights[0].eval()
    b = q_model.trainable_weights[1].eval()
    # print(W,b)
    check_v(pfpo.F_q(s, a), NN_output(s, a, [W, b]))

    print('\n--- checking bellman error')
    berr = pfpo.F_bellman_err(s, a, nexts, r, discrete_actions)
    tv = empirical_bop(s, a, r, nexts, discrete_actions, gamma, NN_output, [W, b])
    check_v(berr, tv, 1)

    print('\n--- checking gradient of the bellman error')
    berr_grad = pfpo.F_grad_bellman_berr(s, a, nexts, r, discrete_actions)
    eps = np.sqrt(np.finfo(float).eps)
    f = lambda x: empirical_bop(s, a, r, nexts, discrete_actions, gamma, NN_output, [x])
    G = np.concatenate((W.ravel(), b.ravel()),axis=0)
    print(G)
    approx_grad = optimize.approx_fprime(G, f, eps)
    for e1, e2 in zip(berr_grad, [np.array(approx_grad[0:2].reshape(2,1)), np.array(approx_grad[2])]):
        check_v(e1, e2, 1)
