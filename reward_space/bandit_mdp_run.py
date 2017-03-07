from __future__ import print_function
from ifqi.evaluation import evaluation
import numpy as np
import ifqi.envs as envs
from policy import BanditPolicy
from ifqi.envs.bandit import Bandit
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

def compute_mu_opt(mdp, n_episodes, discount_factor, mu_min, mu_max, mu_step):
    _range = np.arange(mu_min, mu_max, mu_step)
    grad_J_vec = np.zeros(len(_range))
    print('Finding the best parameter')
    for i,mu in enumerate(_range):
        policy = BanditPolicy(mu)
        n_episodes = 100
        dataset = evaluation.collect_episodes(mdp, policy, n_episodes)
        dataset = add_discount(dataset, 5, discount_factor)
        states = dataset[:, 0]
        discounts = dataset[:, 6]
        rewards = dataset[:, 2]
        PI = policy.get_distribution()
        Q_true = mdp.computeQFunction(policy)
        C = policy.gradient_log_pdf().T
        D = estimate_D(states, discounts)
        W = np.diag(np.diag(D).dot(PI))
        grad_J_true = 1.0 / n_episodes * LA.multi_dot([C.T, W, Q_true])
        J_hat = 1.0 / n_episodes * np.sum(rewards * discounts)
        grad_J_vec[i] = grad_J_true
        print('mu = %f grad_J = %f J=%f' % (mu, grad_J_true,J_hat))
    best = _range[np.argmin(np.abs(grad_J_vec))]
    print('The best parameter is %f' % best)
    return best

mdp = envs.Bandit(0.99)

#MDP parameters
discount_factor = mdp.gamma
horizon = mdp.horizon

#Find best parameter
n_episodes = 100
mu = compute_mu_opt(mdp, n_episodes, discount_factor, 2.1477, 2.1478, 0.00001)
#mu = compute_mu_opt(mdp, n_episodes, discount_factor, 2.4, 2.6, 0.001)

#Policy parameters
policy = BanditPolicy(mu)

#Collect samples
dataset = evaluation.collect_episodes(mdp, policy, n_episodes)
dataset = add_discount(dataset, 5, discount_factor)

PI = policy.get_distribution()

Q_true = mdp.computeQFunction(policy)
print('\nTrue Q function %s\n' % str(Q_true))

states_actions = dataset[:,1]
states = dataset[:,0]
discounts = dataset[:,6]
rewards = dataset[:,2]
D = estimate_D(states, discounts)
W = np.diag(np.diag(D).dot(PI))

C = policy.gradient_log_pdf().T
X = nullspace(C.T.dot(W))

#---------------------------Q-function evaluation-------------------------------

Q_hat, w, rmse = estimate_Q(X, Q_true)
error = np.abs(Q_true - Q_hat)
mae = np.mean(error)
error_rel = np.abs((Q_true - Q_hat)/Q_true)
mare = np.mean(error_rel)
grad_J_true = 1.0/n_episodes * LA.multi_dot([C.T, W, Q_true])
grad_J_hat = 1.0/n_episodes * LA.multi_dot([C.T, W, Q_hat])
J_hat = 1.0/n_episodes * np.sum(rewards * discounts)
print('Estimation of W from samples for states only')
print('Results of LS rmse = %s mae = %s mare = %s' % (rmse, mae, mare))
print('Estimated Q function %s' % Q_hat)
print('True policy gradient %s' % grad_J_true)
print('Estimated policy gradient %s' % grad_J_hat)
print('Estimated expected return %s' % J_hat)
print('Gradient log policy %s' % C.T)