from __future__ import print_function
from ifqi.evaluation import evaluation
import numpy as np
import ifqi.envs as envs
from policy import AdvertisingPolicy, AdvertisingSigmoidPolicy, AdvertisingGaussianPolicy
from ifqi.envs.advertisingMDP import AdvertisingMDP
from utils import add_discount
from operator import itemgetter
from itertools import groupby
from rank_nullspace import nullspace
import numpy.linalg as LA

def estimate_W(state_actions, discounts ):
    '''
    Computes the matrix W, diagonal, having on the diagonal the discounted
    count of the state action pairs
    '''
    dataset = zip(state_actions,discounts)    
    dataset.sort(key=itemgetter(0))
    diag = [sum(zip(*(list(group)))[1]) for key, group in groupby(dataset, key=itemgetter(0))]
    W = np.diag(diag)
    return W
    
def estimate_D(states, discounts):
    '''
    Computes the matrix D, diagonal, having on the diagonal the discounted
    count of the states
    '''
    dataset = zip(states,discounts)    
    dataset.sort(key=itemgetter(0))
    diag = [sum(zip(*(list(group)))[1]) for key, group in groupby(dataset, key=itemgetter(0))]
    D = np.diag(diag)
    return D

def estimate_Q(X, Q_true):
    '''
    Performs LS estimation of the Q function starting from the orthonormal
    basis X and the target Q_true
    '''
    w, residuals, rank, _ =  LA.lstsq(X, Q_true)
    rmse = np.sqrt(residuals/X.shape[0])
    Q_hat = X.dot(w)
    return Q_hat, w, rmse

mdp = envs.AdvertisingMDP(0.99)

#MDP parameters
discount_factor = mdp.gamma
horizon = mdp.horizon

#Policy parameters
policy = AdvertisingGaussianPolicy(eps=0.1, theta1 = 0.1, theta2 = 0.1)

#Collect samples
n_episodes = 100
dataset = evaluation.collect_episodes(mdp, policy, n_episodes)
dataset = add_discount(dataset, 5, discount_factor)

PI = policy.get_distribution()

Q_true = mdp.computeQFunction(policy)
print('True Q function %s\n' % str(Q_true))

states_actions = dataset[:,1]
states = dataset[:,0]
discounts = dataset[:,6]
rewards = dataset[:,2]
W1 = estimate_W(states_actions, discounts)
D = estimate_D(states, discounts)
W2 = np.diag(np.diag(D).dot(PI))

C = policy.gradient_log_pdf().T
X1 = nullspace(C.T.dot(W1))
X2 = nullspace(C.T.dot(W2))

#---------------------------Q-function evaluation 1-----------------------------

Q_hat, w, rmse = estimate_Q(X1, Q_true)
error = np.abs(Q_true - Q_hat)
mae = np.mean(error)
error_rel = np.abs((Q_true - Q_hat)/Q_true)
mare = np.mean(error_rel)
grad_J_true = 1.0/n_episodes * LA.multi_dot([C.T, W1, Q_true])
grad_J_hat = 1.0/n_episodes * LA.multi_dot([C.T, W1, Q_hat])
J_hat = 1.0/n_episodes * np.sum(rewards * discounts)
print('Estimation of W from samples for both states and actions')
print('Results of LS rmse = %s mae = %s mare = %s' % (rmse, mae, mare))
print('Estimated Q function %s' % Q_hat)
print('True policy gradient %s' % grad_J_true)
print('Estimated policy gradient %s' % grad_J_hat)
print('Estimated expected return %s' % J_hat)

#---------------------------Q-function evaluation 2-----------------------------

Q_hat, w, rmse = estimate_Q(X2, Q_true)
error = np.abs(Q_true - Q_hat)
mae = np.mean(error)
error_rel = np.abs((Q_true - Q_hat)/Q_true)
mare = np.mean(error_rel)
grad_J_true = 1.0/n_episodes * LA.multi_dot([C.T, W2, Q_true])
grad_J_hat = 1.0/n_episodes * LA.multi_dot([C.T, W2, Q_hat])
J_hat = 1.0/n_episodes * np.sum(rewards * discounts)
print('\nEstimation of W from samples for states only')
print('Results of LS rmse = %s mae = %s mare = %s' % (rmse, mae, mare))
print('Estimated Q function %s' % Q_hat)
print('True policy gradient %s' % grad_J_true)
print('Estimated policy gradient %s' % grad_J_hat)
print('Estimated expected return %s' % J_hat)
