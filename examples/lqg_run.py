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
print('Dataset has shape {}'.format(dataset.shape))

state_dim, action_dim, reward_dim = envs.get_space_info(mdp)

reward_idx = state_dim + action_dim
nextstate_idx = reward_idx + reward_dim
absorbing_idx = nextstate_idx + state_dim

states = dataset[:, :state_dim]
actions = dataset[:, state_dim:state_dim + action_dim]
rewards = dataset[:, reward_idx:reward_idx + reward_dim]
next_states = dataset[:, nextstate_idx:nextstate_idx+state_dim]
absorbing_flag = dataset[:, absorbing_idx:absorbing_idx+1]
endofepisode_flag = dataset[:, -1]

print()
