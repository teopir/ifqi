from __future__ import print_function
from ifqi.algorithms.pbo.pfpo import PFPO
import numpy as np
import theano
import theano.tensor as T
from time import time
from scipy import optimize


def lqr_reg(s, a, theta):
    theta = theta.reshape(-1,2)
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


def empirical_bop(s, a, r, snext, all_a, gamma, theta):
    qnop = lqr_reg(s, a, theta)
    bop = -np.ones(s.shape[0]) * np.inf
    for i in range(s.shape[0]):
        for j in range(a.shape[0]):
            qv = lqr_reg(snext[i], all_a[j], theta)
            if qv > bop[i]:
                bop[i] = qv
    v = qnop - r - gamma * bop
    return 0.5 * np.mean(v ** 2)


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
            T_s = T.matrix()
            T_a = T.matrix()
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

    s = np.array([1., 2., 3.]).reshape(-1, 1)
    a = np.array([0., 3., 4.]).reshape(-1, 1)
    nexts = (s + 1).copy()
    r = np.array([-1., -5., 0.])
    discrete_actions = np.array([1, 2, 3]).reshape(-1, 1)  # discretization of the actions
    # to be used for maximum estimate

    q_model = LQRRegressor(theta)  # q-function

    pfpo = PFPO(q_model=q_model,
                discrete_actions=discrete_actions,
                gamma=gamma, optimizer="adam",
                state_dim=1, action_dim=1)
    start = time()
    pfpo._make_additional_functions()
    print('compilation time: {}'.format(time() - start))

    check_v(pfpo.F_q(s, a), lqr_reg(s, a, q_model.theta.eval()))

    print('\n--- checking bellman error')
    berr = pfpo.F_bellman_err(s, a, nexts, r, discrete_actions)
    tv = empirical_bop(s, a, r, nexts, discrete_actions, gamma, q_model.theta.eval())
    check_v(berr, tv, 1)

    print('\n--- checking gradient of the bellman error')
    berr_grad = pfpo.F_grad_bellman_berr(s, a, nexts, r, discrete_actions)
    eps = np.sqrt(np.finfo(float).eps)
    f = lambda x: empirical_bop(s, a, r, nexts, discrete_actions, gamma, x)
    approx_grad = optimize.approx_fprime(q_model.theta.eval().ravel(), f, eps).reshape(berr_grad[0].shape)
    check_v(berr_grad, approx_grad, 1)

