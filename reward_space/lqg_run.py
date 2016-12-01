from __future__ import print_function
from ifqi.envs import LQG1D
from ifqi.evaluation import evaluation
import numpy as np
import ifqi.envs as envs

from policy import GaussianPolicy1D, AdvertisingPolicy, AdvertisingSigmoidPolicy
from reward_selector import OrtogonalPolynomialSelector, Orthogonalizer, computeX, SampleGramSchmidt
from utils import add_discount

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
sigma1 = 0.1
sigma2 = 0.01

policy1 = GaussianPolicy1D(K,sigma1,action_bounds)
policy2 = GaussianPolicy1D(K,sigma2,action_bounds)

#Collect samples
dataset1 = evaluation.collect_episodes(mdp, policy=policy1, n_episodes=100)
dataset2 = evaluation.collect_episodes(mdp, policy=policy2, n_episodes=100)

print('Dataset1 (sigma %f) has %d samples' % (sigma1, dataset1.shape[0]))
print('Dataset2 (sigma %f) has %d samples' % (sigma2, dataset2.shape[0]))

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

#---------------------------Q-function evaluation-----------------------------
#Collect data for Q estimation
dataset1_Q = evaluation.collect_episodes(mdp, policy=policy1, n_episodes=100)
states1 = dataset1_Q[:,:state_dim]
actions1 = dataset1_Q[:,state_dim:state_dim+action_dim]
dataset2_Q = evaluation.collect_episodes(mdp, policy=policy2, n_episodes=100)
states2 = dataset2_Q[:,:state_dim]
actions2 = dataset2_Q[:,state_dim:state_dim+action_dim]

def evaluateQChebichev(states, actions, K, sigma, degrees):
    y = np.array(map(lambda s,a: mdp.computeQFunction(s, a, K, sigma), states, actions))
    X = computeX(states, actions, degrees) 
    w, residuals, _, _ =  np.linalg.lstsq(X,y)
    rmse = np.sqrt(residuals/X.shape[0])
    return w, rmse

#print evaluateQChebichev(states1, actions1, K, sigma1, degrees[top5_1])
#print evaluateQChebichev(states2, actions2, K, sigma2, degrees[top5_2])  


from ifqi.envs.advertisingMDP import AdvertisingMDP

mdp = envs.AdvertisingMDP()
policy = AdvertisingSigmoidPolicy(K=[5, 5], eps=0.1)
dataset1 = evaluation.collect_episodes(mdp, policy=policy, n_episodes=100)
dataset1 = add_discount(dataset1, 5, 0.9)

Q = mdp.computeQFunction(policy)

s = SampleGramSchmidt(dataset1[:,1])
matrix = np.hstack([policy.gradient_log_pdf(0,0).T,np.eye(5)])

from itertools import groupby
[(key,len(list(group)), list(group)) for key, group in groupby(sorted(dataset1,key=lambda x: x[1]), key=lambda x: x[1])]

