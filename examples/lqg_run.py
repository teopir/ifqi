from __future__ import print_function
from ifqi.envs import LQG1D
from ifqi.evaluation import evaluation
import numpy as np


class lqr_policy(object):

    def __init__(self, K):
        self.K = K

    def draw_action(self, state, done):
        return np.dot(self.K, state)


mdp = LQG1D()
print(mdp.observation_space)

K = -0.60
# K = mdp.computeOptimalK();

policy = lqr_policy(K)

dataset = evaluation.collect_episodes(mdp, policy=policy, n_episodes=1)
print('Dataset has %d samples' % dataset.shape[0])
