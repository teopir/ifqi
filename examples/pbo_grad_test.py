from __future__ import print_function
from ifqi.algorithms.pbo.gradpbo import GradPBO
import numpy as np
from scipy import optimize


def lqr_reg(s, a, theta):
    w1 = theta[0]
    w2 = theta[1]
    v = - w1 ** 2 * s * a - 0.5 * w2 * (a ** 2) - 0.4 * w2 * (s ** 2)
    return v.ravel()


def grad_lqr_reg(s, a, theta):
    w1 = theta[0]
    w2 = theta[1]
    g1 = -2 * w1 * s * a
    g2 = -0.5 * (a ** 2) - 0.4 * (s ** 2)
    return np.array([g1, g2])


def bellmanop(rho, theta):
    rho_s = np.reshape(rho, (-1, theta.shape[0]))
    return np.dot(rho_s, theta)


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

gamma = 0.99
rho = np.array([1, 2, 10., 3.]).reshape(2, 2)
theta = np.array([2, 0.2]).reshape(-1, 1)

s = np.array([1, 2, 3]).reshape(-1, 1)
a = np.array([0, 3, 4]).reshape(-1, 1)
nexts = s + 1
r = np.array([-1, -5, 0])

gpbo = GradPBO(gamma=gamma)
assert np.allclose(bellmanop(rho, theta), gpbo.bopf(rho, theta)), \
    '{}, {}'.format(bellmanop(rho, theta), gpbo.bopf(rho, theta))
assert np.allclose(lqr_reg(s, a, theta), gpbo.qf(s, a, theta))

actions = np.array([1, 2, 3]).reshape(-1, 1)

berr = gpbo.berrf(s, a, nexts, r, rho, theta, actions)
tv = empirical_bop(s, a, r, nexts, actions, gamma, rho, theta)
assert np.allclose(berr, tv), '{}, {}'.format(berr, tv)

berr_grad = gpbo.grad_berrf(s, a, nexts, r, rho, theta, actions)
eps = np.sqrt(np.finfo(float).eps)
f = lambda x: empirical_bop(s, a, r, nexts, actions, gamma, x, theta)
approx_grad = optimize.approx_fprime(rho.ravel(), f, eps).reshape(berr_grad.shape)
assert np.allclose(berr_grad, approx_grad), '{}, {}'.format(berr_grad, approx_grad)