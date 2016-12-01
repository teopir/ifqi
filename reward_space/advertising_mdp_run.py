from __future__ import print_function
from ifqi.evaluation import evaluation
import numpy as np
import ifqi.envs as envs
from policy import AdvertisingPolicy, AdvertisingSigmoidPolicy
from ifqi.envs.advertisingMDP import AdvertisingMDP
from utils import add_discount
from operator import itemgetter
from itertools import groupby
from rank_nullspace import nullspace

def estimateW(state_actions, discounts ):
    dataset = zip(state_actions,discounts)    
    dataset.sort(key=itemgetter(0))
    diag = [sum(zip(*(list(group)))[1]) for key, group in groupby(dataset, key=itemgetter(0))]
    W = np.diag(diag)
    return W
    
def estimateD(states, discounts):
    dataset = zip(states,discounts)    
    dataset.sort(key=itemgetter(0))
    diag = [sum(zip(*(list(group)))[1]) for key, group in groupby(dataset, key=itemgetter(0))]
    D = np.diag(diag)
    return D


mdp = envs.AdvertisingMDP(gamma = 0.99)

policy = AdvertisingSigmoidPolicy(K=[5, 5], eps=0.1)
dataset1 = evaluation.collect_episodes(mdp, policy=policy, n_episodes=100)
dataset1 = add_discount(dataset1, 5, mdp.gamma)

PI = policy.get_distribution()

Q_true = mdp.computeQFunction(policy)
print('True Q function %s' % str(Q_true))

W1 = estimateW(dataset1[:,1],dataset1[:,6])
D = estimateD(dataset1[:,0], dataset1[:,6])
W2 = np.diag(np.diag(D).dot(PI))

policy_gradient = policy.gradient_log_pdf()
B1 = nullspace(policy_gradient.dot(W1))
B2 = nullspace(policy_gradient.dot(W2))

w1, r1, _, _ = np.linalg.lstsq(B1,Q_true)
rmse1 = np.sqrt(r1/Q_true.shape[0])
Q_hat1 = B1.dot(w1)
print('Q function estimated without policy %s' % str(Q_hat1))
print('RMSE1 %f' % rmse1)

w2, r2, _, _ = np.linalg.lstsq(B2,Q_true)
rmse2 = np.sqrt(r2/Q_true.shape[0])
Q_hat2 = B2.dot(w2)
print('Q function estimated policy %s' % str(Q_hat2))
print('RMSE2 %f' % rmse2)