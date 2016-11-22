from __future__ import print_function
from ifqi.envs import LQG1D
from ifqi.evaluation import evaluation
import numpy as np
import matplotlib.pyplot as plt
import ifqi.envs as envs

from policy import GaussianPolicy1D
from reward_selector import OrtogonalPolynomialSelector

mdp = LQG1D()

#MDP parameters
discount_factor = mdp.gamma
horizon = mdp.horizon
max_action = mdp.max_action
max_pos = mdp.max_pos
state_dim, action_dim, reward_dim = envs.get_space_info(mdp)

#Policy parameters
action_bounds = np.array([[-max_action], [max_action]], ndmin=2)
state_bounds = np.array([[-max_pos] , [max_pos]], ndmin=2)
K = -0.60

policy1 = GaussianPolicy1D(K,0.1,action_bounds)
policy2 = GaussianPolicy1D(K,0.01,action_bounds)

#Collect samples
dataset1 = evaluation.collect_episodes(mdp, policy=policy1, n_episodes=500)
dataset2 = evaluation.collect_episodes(mdp, policy=policy2, n_episodes=500)

print('Dataset1 (sigma %f) has %d samples' % (0.1, dataset1.shape[0]))
print('Dataset2 (sigma %f) has %d samples' % (0.01, dataset2.shape[0]))

endofepisode_idx = -1

state_actions1 = dataset1[:,:state_dim + action_dim]
state_actions2 = dataset2[:,:state_dim + action_dim] 

bounds = [[-max_pos, max_pos], [-max_action ,max_action]]

max_degree = 10
ops = OrtogonalPolynomialSelector(ndim=2, bounds=bounds, max_degree = [max_degree,max_degree])
ort1 = ops.compute(state_actions1,policy1)
ort2 = ops.compute(state_actions2,policy2)

degrees = np.array([(x,y) for x in range(max_degree+1) for y in range(max_degree+1)])


top5_1 = np.abs(ort1).argsort(axis=None)[:5]
top5_2 = np.abs(ort2).argsort(axis=None)[:5]

print('\nTop 3 ortogonal chebichev polys for Dataset1 ([state_degree, action_degree], scalar_product)')
print(zip(degrees[top5_1].tolist(), ort1.ravel()[top5_1]))

print('\nTop 3 ortogonal chebichev polys for Dataset2 ([state_degree, action_degree], scalar_product)')
print(zip(degrees[top5_2].tolist(), ort2.ravel()[top5_2]))
