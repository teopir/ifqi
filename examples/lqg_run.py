from __future__ import print_function
import ifqi.envs as envs
from ifqi.evaluation import evaluation
import numpy as np


class lqr_policy(object):
    def __init__(self, K):
        self.K = K

    def draw_action(self, state, done):
        return np.dot(self.K, state)


mdp = envs.LQG1D()
print(mdp.observation_space)

K = -0.60
# K = mdp.computeOptimalK();

policy = lqr_policy(K)

dataset = evaluation.collect_episodes(mdp, policy=policy, n_episodes=20)
print('Dataset has %d samples' % dataset.shape[0])

state_dim, action_dim = envs.get_space_info(mdp)
reward_dim = 1

states = dataset[:, 0:state_dim]
actions = dataset[:, state_dim:state_dim + action_dim]
rewards = dataset[:, state_dim + action_dim:state_dim + action_dim + reward_dim]
next_states = dataset[:, state_dim + action_dim + reward_dim:-1]
absorbing_flag = dataset[:, -1]

print()
