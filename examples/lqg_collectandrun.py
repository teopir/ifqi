from __future__ import print_function
import numpy as np
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
pol = tmp_policy(K, S)

