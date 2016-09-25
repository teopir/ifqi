from __future__ import print_function
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath('../'))
from context import *

from ifqi.envs.lqg1d import LQG1D

env = LQG1D()
K = env.computeOptimalK()
print('Optimal K: {}'.format(K))
S = 0.5  # covariance of the controller
print('Covariance S: {}'.format(S))


class tmp_policy():
    def __init__(self, M, S):
        self.K = M
        if isinstance(S, (int, float)):
            self.S = np.array([S]).reshape(1, 1)

    def drawAction(self, state):
        return np.dot(self.K, state) + np.random.multivariate_normal(
            np.zeros(self.S.shape[0]), self.S, 1)

##############################################################
# Compute the discounted reward
# n_rep = 1000
# J = env.computeJ(K, S, n_random_x0=n_rep)
pol = tmp_policy(K, S)
# Jsample = env.evaluate(pol, nbEpisodes=n_rep, metric='discounted', render=False)
# print(J, Jsample)

##############################################################
# Collect samples
A = env.collectEpisode(None)
print(A.shape)
