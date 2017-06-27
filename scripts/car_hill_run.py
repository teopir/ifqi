import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor

from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_data_for_fqi
from ifqi.algorithms.fqi import FQI
from ifqi.models.actionregressor import ActionRegressor
from ifqi.models.regressor import Regressor
from ifqi.models.mlp import MLP
from ifqi.models.ensemble import Ensemble


from reward_space.utils.continuous_env_sample_estimator import ContinuousEnvSampleEstimator
from reward_space.policy_gradient.gradient_estimator import MaximumLikelihoodEstimator
import reward_space.utils.linalg2 as la2
import numpy.linalg as la


def fit_maximum_likelihood_policy_from_trajectories(  trajectories,
                                                      policy,
                                                      initial_parameter,
                                                      max_iter=100,
                                                      learning_rate=0.01):
    n_trajectories = int(trajectories[:, -1].sum())
    n_samples = trajectories.shape[0]

    parameter = initial_parameter
    ite = 0
    i = 0
    while ite < max_iter:
        ite += 1
        policy.set_parameter(parameter, build_hessian=False)

        #Gradient computation

        gradient = 0.
        while trajectories[i, -1] == 0:
            gradient += policy.gradient_log(trajectories[i, :2], trajectories[i, 2])
            print(la.norm(gradient))
            i += 1
        gradient += policy.gradient_log(trajectories[i, :2], trajectories[i, 2])
        gradient /= -n_trajectories
        gradient = gradient.ravel()[:, np.newaxis]
        i = (i + 1) % n_samples

        parameter = parameter - learning_rate * gradient

    policy.set_parameter(parameter)
    return policy

mdp = envs.CarOnHill()





state_dim, action_dim, reward_dim = envs.get_space_info(mdp)
assert reward_dim == 1

regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'input_scaled': False,
                    'output_scaled': False}
discrete_actions = mdp.action_space.values

# ExtraTrees
regressor = Regressor(ExtraTreesRegressor, **regressor_params)

# Action regressor of Ensemble of ExtraTreesEnsemble
# regressor = Ensemble(ExtraTreesRegressor, **regressor_params)
regressor = ActionRegressor(regressor, discrete_actions=discrete_actions,
                            tol=5)

dataset = evaluation.collect_episodes2(mdp, n_episodes=1000)
check_dataset(dataset, state_dim, action_dim, reward_dim) # this is just a
# check, it can be removed in experiments
print('Dataset has %d samples' % dataset.shape[0])



# reward_idx = state_dim + action_dim
# sast = np.append(dataset[:, :reward_idx],
#                  dataset[:, reward_idx + reward_dim:-1],
#                  axis=1)
# r = dataset[:, reward_idx]
sast, r = split_data_for_fqi(dataset, state_dim, action_dim, reward_dim)

fqi_iterations = mdp.horizon  # this is usually less than the horizon
fqi = FQI(estimator=regressor,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=mdp.gamma,
          horizon=fqi_iterations,
          verbose=True)

fit_params = {}

fqi.partial_fit(sast, r, **fit_params)

#ex_returns = []
#n_episodes_expert = 20
#trajectories_expert = evaluation.collect_episodes(mdp, fqi, n_episodes_expert)
#ex_return = np.dot(trajectories_expert[:, 3], trajectories_expert[:, 6])
#ex_returns.append(ex_return)

iterations = 20
iteration_values = []

for i in range(iterations - 1):
    fqi.partial_fit(None, None, **fit_params)

    #values = evaluation.evaluate_policy(mdp, fqi, initial_states=mdp.initial_states)

    #Collect episodes from expert's policy
    #n_episodes_expert = 20
    #trajectories_expert = evaluation.collect_episodes(mdp, fqi, n_episodes_expert)
    #ex_return = np.dot(trajectories_expert[:, 3], trajectories_expert[:, 6])
    #print('Expert trajectories have %d samples' % trajectories_expert.shape[0])
    #print('Expert return %s' % ex_return)
    #ex_returns.append(ex_return)

e=0.9
class epsilon_expert(object):

    def draw_action(self, state, absorbing=False):
        action = fqi.draw_action(state, absorbing)
        if np.random.uniform() > e:
            return 4 if np.random.randint(2) == 0 else -4
        else:
            return action

r = []
#for i in range(20):
n_episodes_expert = 20
trajectories_expert = evaluation.collect_episodes(mdp, epsilon_expert(), n_episodes_expert)
ex_return = np.dot(trajectories_expert[:, 3], trajectories_expert[:, 6])
print('Expert trajectories have %d samples' % trajectories_expert.shape[0])
print('Expert return %s' % (ex_return/20.0))


n_dim_centers = 20
n_centers = n_dim_centers * n_dim_centers
centers = np.meshgrid(np.linspace(-mdp.max_pos, mdp.max_pos, n_dim_centers),
                      np.linspace(-mdp.max_velocity, mdp.max_velocity, n_dim_centers))

centers = np.vstack([centers[0].ravel(), centers[1].ravel()]).T

parameters = np.zeros((n_centers, 1))
from policy import RBFGaussianPolicy
policy = RBFGaussianPolicy(centers, parameters, sigma=0.01, radial_basis_parameters=0.01)

#DA SISTEMARE
def fit_lr(trajectories, centers, policy):
    states = trajectories[:, :2]
    actions = trajectories[:, 2]
    X = np.zeros((len(states), len(centers)))
    for i in range(len(states)):
        for j in range(len(centers)):
            X[i, j] = policy.radial_basis(states[i], centers[j])

    from sklearn.linear_model import Ridge
    lr = Ridge(alpha=0.0005, fit_intercept=False)
    lr.fit(X, actions)

    print(lr.coef_)
    policy.set_parameter(lr.coef_[:, np.newaxis])
    return policy

ml_policy = fit_lr(trajectories_expert, centers, policy)

n_episodes_ml = 20
trajectories_ml = evaluation.collect_episodes(mdp, ml_policy, n_episodes_ml)
print('ML policy trajectories have %d samples' % trajectories_ml.shape[0])
print('ML policy return %s' % np.dot(trajectories_ml[:, 3], trajectories_ml[:, 6]/20))
r.append(np.dot(trajectories_ml[:, 3], trajectories_ml[:, 6]/20))

rmean = np.mean(r)
rstd = np.std(r)

import scipy.stats as st
rci = st.t.interval(0.9, 20-1, loc=rmean, \
                            scale=rstd/np.sqrt(20-1))

print(rmean)
print(rci)

d_sa_mu_hat = trajectories_expert[:, 6]
D_hat = np.diag(d_sa_mu_hat)

states_actions = trajectories_expert[:, :3]
states = trajectories_expert[:, :2]
actions = trajectories_expert[:, 2]
rewards = trajectories_expert[:, 3]
n_samples = trajectories_expert.shape[0]

G = np.array(map(lambda s,a: policy.gradient_log(s,a), states, actions))
H = np.array(map(lambda s,a: policy.hessian_log(s,a), states, actions))

print('-' * 100)
print('Computing Q-function approx space...')

X = np.dot(G.T, D_hat)
phi = la2.nullspace(X)

print('-' * 100)
print('Computing A-function approx space...')

sigma_kernel = 0.01
def gaussian_kernel(x):
    return 1. / np.sqrt(2 * np.pi * sigma_kernel ** 2) * np.exp(
        - 1. / 2 * x ** 2 / (sigma_kernel ** 2))

from sklearn.neighbors import NearestNeighbors
knn_states = NearestNeighbors(n_neighbors=10)
knn_states.fit(states)
pi_tilde = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    pi_tilde[i, i] = policy.pdf(states[i], actions[i])
    _, idx = knn_states.kneighbors([states[i]])
    idx = idx.ravel()
    for j in idx:
        pi_tilde[i, j] = policy.pdf(states[j], actions[j])

pi_tilde /= pi_tilde.sum(axis=1)
pi_tilde[np.isnan(pi_tilde)] = 0.

Y = np.dot(np.eye(n_samples) - pi_tilde, phi)
psi = la2.range(Y)

names = []
basis_functions = []
gradients = []
hessians = []

names.append('Reward function')
basis_functions.append(rewards / la.norm(rewards))

ml_estimator = MaximumLikelihoodEstimator(trajectories_expert)
gradient_hat = ml_estimator.estimate_gradient(psi, G, use_baseline=True)
hessian_hat = ml_estimator.estimate_hessian(psi, G, H, use_baseline=True)

eigval_hat, _ = la.eigh(hessian_hat)
eigmax_hat, eigmin_hat = eigval_hat[:, -1], eigval_hat[:, 0]
trace_hat = np.trace(hessian_hat, axis1=1, axis2=2)

# Heuristic for negative semidefinite
neg_idx = np.argwhere(eigmax_hat < -eigmin_hat).ravel()
hessian_hat_neg = hessian_hat[neg_idx]

from reward_space.inverse_reinforcement_learning.hessian_optimization import HeuristicOptimizerNegativeDefinite
optimizer = HeuristicOptimizerNegativeDefinite(hessian_hat_neg)
w = optimizer.fit(skip_check=True)
eco_r = np.dot(psi[:, neg_idx], w)

names.append('ECO-R')
basis_functions.append(eco_r)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaled_basis_functions = []
for bf in basis_functions:
    sbf = scaler.fit_transform(bf[:, np.newaxis]).ravel()
    scaled_basis_functions.append(sbf)

print('Estimating gradient and hessians...')
gradients, hessians, gradient_norms2, gradient_normsinf, eigvals, traces, eigvals_max = [], [], [], [], [], [], []
print('Estimating gradient and hessians...')
for sbf in scaled_basis_functions:
    gradient = ml_estimator.estimate_gradient(sbf, G, use_baseline=True)[0]
    hessian = ml_estimator.estimate_hessian(sbf, G, H, use_baseline=True)[0]
    gradients.append(gradient)
    hessians.append(hessian)
    eigval, _ = la.eigh(hessian)
    gradient_norms2.append(la.norm(gradient, 2))
    gradient_normsinf.append(la.norm(gradient, np.inf))
    eigvals_max.append(eigval[-1])
    eigvals.append(eigval)
    traces.append(np.trace(hessian))

from prettytable import PrettyTable
t = PrettyTable()
t.add_column('Basis function', names)
t.add_column('Gradient norm 2', gradient_norms2)
t.add_column('Gradient norm inf', gradient_normsinf)
t.add_column('Hessian  max eigenvalue', eigvals_max)
t.add_column('Hessian trace', traces)
print(t)

estimator = ContinuousEnvSampleEstimator(trajectories_expert, mdp.gamma)
count_sa_hat = estimator.get_count_sa()
count_sa_hat /= count_sa_hat.max()

from reward_space.utils.k_neighbors_regressor_2 import KNeighborsRegressor2
count_sa_knn = KNeighborsRegressor2(n_neighbors=5, weights=gaussian_kernel)
count_sa_knn.fit(states_actions, count_sa_hat)
max_value = count_sa_knn.predict(states_actions, rescale=False).max()

import copy
mdp2 = copy.deepcopy(mdp)

from sklearn.neighbors import KNeighborsRegressor

k=5
knn = KNeighborsRegressor(n_neighbors=k, weights=gaussian_kernel)
knn.fit(states_actions, sbf)

def reward_function(self, state, action):
    return 0.5 * self.knn.predict([np.hstack([state, [action]])]) + \
           0.5 / self.max_value * self.count_sa_knn.predict([np.hstack([state, [action]])], rescale=False)

def reward_function2(state, action):
    return 0.5 * knn.predict([np.hstack([state, [action]])]) + \
           0.5 / max_value * count_sa_knn.predict([np.hstack([state, [action]])], rescale=False)

absorbing_value = np.mean(sbf[np.argwhere(trajectories_expert[:, -1] == 1.)])

from scipy.integrate import odeint
def new_step(self, u):
    sa = np.append(self._state, u)
    new_state = odeint(self._dpds, sa, [0, self._dt])

    self._state = new_state[-1, :-1]

    if self._state[0] < -self.max_pos or \
                    np.abs(self._state[1]) > self.max_velocity:
        self._absorbing = True
        reward = -1
    elif self._state[0] > self.max_pos and \
                    np.abs(self._state[1]) <= self.max_velocity:
        self._absorbing = True
        reward = 1
    else:
        reward = 0

    if self._state[0] > self.max_pos and \
                    np.abs(self._state[1]) <= self.max_velocity:
        reward = (-0.95 ** 100 + 0.95 ** 20) / (1 - 0.95) * self.absorbing_value
    else:
        reward = self.reward_function(self._state, np.asscalar(u))
        if np.isnan(reward):
            reward=0.

    return self.get_state(), reward, self._absorbing, {}

import types

mdp2.count_sa_knn = count_sa_knn
mdp2.knn = knn
mdp2.absorbing_value = absorbing_value
mdp2.max_value = max_value
mdp2.reward_function = types.MethodType(reward_function, mdp2)
mdp2.step = types.MethodType(new_step, mdp2)


#-------------------------------------------------------------------------------
'''
from reward_space.policy_gradient.policy_gradient_learner import PolicyGradientLearner
iterations=5
learner = PolicyGradientLearner(mdp2, policy, lrate=0.008, verbose=1,
                                    max_iter_opt=iterations, max_iter_eval=100, tol_opt=-1., tol_eval=0.,
                                    estimator='reinforce',
                                    gradient_updater='adam')

theta0 = np.zeros((n_centers,1))
theta, history = learner.optimize(theta0,return_history=True)

policy.set_parameter(theta)
n_episodes_ml = 20
trajectories_ml = evaluation.collect_episodes(mdp, policy, n_episodes_ml)
print('ECO %s' % np.dot(trajectories_ml[:, 3],
                                     trajectories_ml[:, 6] / 20))

learner = PolicyGradientLearner(mdp, policy, lrate=0.008, verbose=1,
                                    max_iter_opt=iterations, max_iter_eval=100, tol_opt=-1., tol_eval=0.,
                                    estimator='reinforce',
                                    gradient_updater='adam')

theta0 = np.zeros((n_centers,1))
theta, history = learner.optimize(theta0,return_history=True)

policy.set_parameter(theta)
n_episodes_ml = 20
trajectories_ml = evaluation.collect_episodes(mdp, policy, n_episodes_ml)
print('Reward %s' % np.dot(trajectories_ml[:, 3],
                                     trajectories_ml[:, 6] / 20))


'''
#-------------------------------------------------------------------------------
#FQI train
regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'input_scaled': False,
                    'output_scaled': False}
discrete_actions = mdp2.action_space.values

# ExtraTrees
regressor = Regressor(ExtraTreesRegressor, **regressor_params)

# Action regressor of Ensemble of ExtraTreesEnsemble
# regressor = Ensemble(ExtraTreesRegressor, **regressor_params)
regressor = ActionRegressor(regressor, discrete_actions=discrete_actions,
                            tol=5)

dataset = evaluation.collect_episodes2(mdp2, n_episodes=1000)
check_dataset(dataset, state_dim, action_dim, reward_dim) # this is just a
# check, it can be removed in experiments
print('Dataset has %d samples' % dataset.shape[0])

# reward_idx = state_dim + action_dim
# sast = np.append(dataset[:, :reward_idx],
#                  dataset[:, reward_idx + reward_dim:-1],
#                  axis=1)
# r = dataset[:, reward_idx]
sast, r = split_data_for_fqi(dataset, state_dim, action_dim, reward_dim)

fqi_iterations = mdp2.horizon  # this is usually less than the horizon
fqi = FQI(estimator=regressor,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=mdp2.gamma,
          horizon=fqi_iterations,
          verbose=True)

fit_params = {}

fqi.partial_fit(sast, r, **fit_params)
'''
lr_returns = []
trajectories_expert2 = evaluation.collect_episodes(mdp, fqi,
                                                   n_episodes_expert)
lr_return = np.dot(trajectories_expert2[:, 3],
                   trajectories_expert2[:, 6])
lr_returns.append(lr_return)
'''
iterations = 20
iteration_values = []

for i in range(iterations - 1):
    fqi.partial_fit(None, None, **fit_params)
    '''
    trajectories_expert2 = evaluation.collect_episodes(mdp, fqi,
                                                      n_episodes_expert)
    lr_return = np.dot(trajectories_expert2[:, 3],
                                      trajectories_expert2[:, 6])
    lr_returns.append(lr_return)
    print('Expert trajectories have %d samples' % trajectories_expert2.shape[0])
    print('Expert return %s' % lr_return)

    '''
import time
mytime = time.time()

#np.save('data/ch/expert_returns_%s' % mytime, np.array(ex_returns))
#np.save('data/ch/eco_returns_explo_%s' % mytime, np.array(lr_returns))


trajectories_expert2 = evaluation.collect_episodes(mdp, fqi,
                                                      n_episodes_expert)


l = [trajectories_expert, trajectories_ml, trajectories_expert2]
np.save('data/coh/%s_%s' % (e,time.time()), l)
'''
fig, ax = plt.subplots()
ax.set_xlabel('p')
ax.set_ylabel('v')
fig.suptitle('Trajectories')

import scipy.stats as st
confidence=0.95

stop = np.argwhere(trajectories_expert[:,-1] == 1.).ravel()+1
start = np.array([0] + list(stop[:-1]))

A = trajectories_expert[:, 3]* trajectories_expert[:, 6]
indices = np.stack([start, stop]).T
c = np.r_[0, A.cumsum()][indices]
sums = c[:,1] - c[:,0]

amax, amin = np.argmax(sums), np.argmin(sums)

tr_ex_best, tr_exp_worst = trajectories_expert[start[amax]:stop[amax], :], trajectories_expert[start[amin]:stop[amin], :]

out = []

for dataset, c, l in zip([trajectories_expert, trajectories_expert2, trajectories_ml], ['r', 'b', 'g'], ['Expert', 'CR-IRL', 'BC']):


    index = np.array([0] + list(np.argwhere(dataset[:,-1] == 1.).ravel() + 1))
    n = len(index)-1
    lmax = np.diff(index).max()+1

    states = np.zeros((n, 2, lmax))
    episode = 0
    t = 0
    for i in range(len(dataset)):
        states[episode, :, t] = dataset[i, :2]
        if dataset[i, -1] == 1.:
            states[episode, :, t+1] = dataset[i, 4:6]
            for tt in range(t+2, lmax):
                states[episode, :, tt] = states[episode, :, t+1]
            episode += 1
            t = 0
        else:
            t += 1

'''
    #p_mean, v_mean = np.mean(states, axis=0)
    #p_std, v_std= np.std(states, axis=0)

    #p_error = st.t.interval(confidence, n - 1, loc=p_mean, \
    #              scale=p_std / np.sqrt(n - 1))[0] - p_mean
    #v_error = st.t.interval(confidence, n - 1, loc=v_mean, \
    #                        scale=v_std / np.sqrt(n - 1))[0] - v_mean
'''
    out.append(states)
    for i in range(n):
        ax.plot(states[i, 0, :], states[i, 1, :], color=c)
    #ax.plot(p_mean, v_mean, color = c, label=l, marker='*')
    #ax.fill_betweenx(v_mean, p_mean-p_error, p_mean+p_error, color = c, alpha=0.5)
    #ax.fill_between(p_mean, v_mean - v_error, v_mean + v_error, color = c,alpha=0.5)
    #ax.errorbar(p_mean, v_mean, xerr=p_std, yerr=v_sdt)
'''
#fig.legend()