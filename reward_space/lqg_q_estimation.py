from __future__ import print_function
from ifqi.envs import LQG1D
from ifqi.evaluation import evaluation
import numpy as np
import ifqi.envs as envs

from policy import GaussianPolicy1D
from reward_selector import OrthogonalPolynomialSelector
from utils import add_discount, chebvalNd

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

#--------------------------Find the most orthogonal polys-----------------------
bounds = [[-max_pos, max_pos], [-max_action ,max_action]]
max_degree = [10,10]
top_n=20

def find_orthogonal_poly(max_degree, bounds, state_actions, policy, top_n=5):
    ops = OrthogonalPolynomialSelector(ndim=len(max_degree), bounds=bounds, max_degree = max_degree)
    ort = ops.compute(state_actions,policy)
    degrees = np.array([(x,y) for x in range(max_degree[0]+1) for y in range(max_degree[1]+1)])
    top_n_ort = np.abs(ort).argsort(axis=None)[:top_n]
    top_n_poly = degrees[top_n_ort]
    return top_n_poly, top_n_ort, ort

top_n_poly1, top_n_ort1, ort1 = find_orthogonal_poly(max_degree, bounds, state_actions1, policy1, top_n)    
top_n_poly2, top_n_ort2, ort2 = find_orthogonal_poly(max_degree, bounds, state_actions2, policy2, top_n) 

print('Dataset1 best degrees are %s' % str(top_n_poly1))
print('Dataset2 best degrees are %s' % str(top_n_poly2))

#---------------------------Q-function evaluation-----------------------------
#Collect data for Q estimation
dataset1_Q = evaluation.collect_episodes(mdp, policy=policy1, n_episodes=100)
states1 = dataset1_Q[:,:state_dim]
actions1 = dataset1_Q[:,state_dim:state_dim+action_dim]
dataset2_Q = evaluation.collect_episodes(mdp, policy=policy2, n_episodes=100)
states2 = dataset2_Q[:,:state_dim]
actions2 = dataset2_Q[:,state_dim:state_dim+action_dim]

Qtrue1 = np.array(map(lambda s,a: mdp.computeQFunction(s, a, K, np.sqrt(sigma1)), states1, actions1))
Qtrue2 = np.array(map(lambda s,a: mdp.computeQFunction(s, a, K, np.sqrt(sigma2)), states2, actions2))

#Computes the feature matrix
def computeX(states, actions, degrees):
    n_samples = states.shape[0]
    n_features = degrees.shape[0]
    X = np.zeros(shape=(n_samples,n_features))
    for j,(ds,da) in enumerate(degrees):
        for i,(s,a) in enumerate(zip(states,actions)):
            X[i,j] = chebvalNd([s,a], [ds,da])
    return X

def evaluateQChebichev(states, actions, K, sigma, degrees, Qtrue):
    X = computeX(states, actions, degrees) 
    print('rank of X %d' % np.linalg.matrix_rank(X))
    w, residuals, rank, _ =  np.linalg.lstsq(X,Qtrue)
    rmse = np.sqrt(residuals/X.shape[0])
    return w, rmse, rank

w1, rmse1, rank1 = evaluateQChebichev(states1, actions1, K, np.sqrt(sigma1), top_n_poly1, Qtrue1)
w2, rmse2, rank2 = evaluateQChebichev(states2, actions2, K, np.sqrt(sigma2), top_n_poly2, Qtrue2)  

print('Results of LS 1 %s %s %s' % (w1, rmse1, rank1))
print('Results of LS 2 %s %s %s' % (w2, rmse2, rank2))