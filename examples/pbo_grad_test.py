from __future__ import print_function
from ifqi.algorithms.pbo.gradpbo import GradPBO
import numpy as np


def lqr_reg(s, a, theta):
    w1 = theta[0]
    w2 = theta[1]
    return - w1 ** 2 * s * a - 0.5 * w2 * (a ** 2) - 0.4 * w2 * (s ** 2)


def grad_lqr_reg(s, a, theta):
    w1 = theta[0]
    w2 = theta[1]
    g1 = -2 * w1 * s * a
    g2 = -0.5 * (a ** 2) - 0.4 * (s ** 2)
    return np.array([g1, g2])


def bellmanop(rho):
    return rho


def bellmanop_grad(rho):
    return np.ones_like(rho)


rho = np.array([1, 2]).reshape(-1,1)
theta = np.array([2, 0.]).reshape(-1, 1)

s = np.array([1, 2, 3]).reshape(-1, 1)
a = np.array([0, 3, 4]).reshape(-1, 1)
nexts = s + 1
r = np.array([-1, -5, 0])

gpbo = GradPBO()
assert np.allclose(bellmanop(rho), gpbo.bopf(rho)),\
    '{}, {}'.format(bellmanop(rho), gpbo.bopf(rho))
assert np.allclose(lqr_reg(s, a, theta), gpbo.qf(s, a, theta))

actions = np.array([1, 2, 3]).reshape(-1, 1)
vv = gpbo.berrf(s, a, nexts, r, rho, theta, actions)
print(vv)

# actions = np.matlib.repmat(np.array([1, 2, 3]).reshape(1, -1), s.shape[0], 1)
# print(actions)
# vv = gpbo.qf(s, actions, theta)
# print(vv)
