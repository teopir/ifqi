from __future__ import print_function

import os
import sys
sys.path.insert(0, os.path.abspath('../'))
from context import *

import numpy as np

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from ifqi.envs.lqg1d import LQG1D

mdp = LQG1D()
K = mdp.computeOptimalK()
print('Optimal K: {}'.format(K))


class tmp_policy():
    def __init__(self, M):
        self.K = M

    def predict(self, state):
        return np.dot(self.K, state), 0


##############################################################
# Compute the discounted reward
n_rep = 1000
J = mdp.computeJ(K, 1e-10, n_random_x0=n_rep)
pol = tmp_policy(K)
Jsample = 0
for i in range(n_rep):
    Jsample += mdp.runEpisode(pol, False, True)[0]
Jsample /= n_rep
print(J, Jsample)

xs = np.linspace(-mdp.max_pos, mdp.max_pos, 60)
us = np.linspace(-mdp.max_action, mdp.max_action, 50)

l = []
for x in xs:
    for u in us:
        v = mdp.computeQFunction(x, u, K, 0.0001, 100)
        l.append([x, u, v])
tabular_Q = np.array(l)
np.savetxt('lqg_qtable.txt', tabular_Q)
print('printed Q-table')

# print(tabular_Q.shape)
# print(tabular_Q)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(tabular_Q[:, 0], tabular_Q[:, 1], tabular_Q[:, 2])
plt.show()
