from __future__ import print_function
from ifqi.envs import LQG1D
from ifqi.evaluation import evaluation
import numpy as np
import matplotlib.pyplot as plt
from gym.utils import seeding
import ifqi.envs as envs

from policy import SimplePolicy, GaussianPolicy1D
from utils import EpisodeIterator

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

policy1 = SimplePolicy(K,action_bounds)
policy2 = GaussianPolicy1D(K,0.1,action_bounds)
policy3 = GaussianPolicy1D(K,0.01,action_bounds)

#Collect samples
dataset1 = evaluation.collect_episodes(mdp, policy=policy1, n_episodes=100)
dataset2 = evaluation.collect_episodes(mdp, policy=policy2, n_episodes=100)
dataset3 = evaluation.collect_episodes(mdp, policy=policy3, n_episodes=100)
print('Dataset1 has %d samples' % dataset1.shape[0])
print('Dataset2 has %d samples' % dataset2.shape[0])
print('Dataset3 has %d samples' % dataset3.shape[0])

reward_idx = state_dim + action_dim
nextstate_idx = reward_idx + reward_dim
absorbing_idx = nextstate_idx + state_dim

states = dataset1[:, :state_dim]
actions = dataset1[:, state_dim:state_dim + action_dim]
rewards = dataset1[:, reward_idx:reward_idx + reward_dim]
next_states = dataset1[:, nextstate_idx:nextstate_idx+state_dim]
absorbing_flag = dataset1[:, absorbing_idx:absorbing_idx+1]
endofepisode_flag = dataset1[:, -1]

ite = EpisodeIterator(dataset1,-1)
first_episode = ite.next()
first_episode_states = first_episode[:, :state_dim]
first_episode_actions = first_episode[:, state_dim:state_dim + action_dim]

#Some plot: one episode
fig = plt.figure()
plt.plot(first_episode_states)
plt.plot(first_episode_actions)



#Consider polocy 2
samples3 = dataset3[:,:state_dim + action_dim] 
samples2 = dataset2[:,:state_dim + action_dim]
bounds = [[-max_pos, max_pos], [-max_action ,max_action]]

from reward_selector import RewardSelector
rs = RewardSelector(ndim=2, bounds=bounds, max_degree = 3)
ort2 = rs.compute(samples2,policy2)
ort3 = rs.compute(samples3,policy3)

angles2 = 1-np.abs(np.pi/2 - np.arccos(ort2))
angles3 = 1-np.abs(np.pi/2 - np.arccos(ort3))
angles2_flat = angles2.ravel()
angles3_flat = angles3.ravel()
angles2_flat.sort()
angles3_flat.sort()
fig = plt.figure()
plt.plot(angles2)
plt.plot(angles3)

from mpl_toolkits.mplot3d import Axes3D
x, y = zip(*[(x,y) for x in range(4) for y in range(4)])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, angles2_flat, c='r', marker='o')
ax.scatter(x, y, angles3_flat, c='b', marker='^')
ax.set_xlabel('state degree')
ax.set_ylabel('action degree')
ax.set_zlabel('ort')
plt.show()

'''
#Computation of real Q
states = np.linspace(-10,10,100)
actions = np.linspace(-8,8,100)
states_v, actions_v = np.meshgrid(states,actions)
q = np.ndarray(states_v.shape)
for index, _ in np.ndenumerate(states_v):
    q[index] = mdp.computeQFunction(states_v[index],actions_v[index],K,0.1)
    

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(states_v, actions_v, q)
'''