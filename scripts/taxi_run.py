from __future__ import print_function
from ifqi.envs import TaxiEnv
from ifqi.evaluation import evaluation
from policy import BoltzmannPolicy, TaxiEnvPolicy
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from reward_space.utils.discrete_env_sample_estimator import DiscreteEnvSampleEstimator
from reward_space.utils.discrete_mdp_wrapper import DiscreteMdpWrapper
from reward_space.proto_value_functions.proto_value_functions_estimator import  ProtoValueFunctionsEstimator
import reward_space.utils.linalg2 as la2
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from reward_space.policy_gradient.gradient_estimator import MaximumLikelihoodEstimator
#import cvxpy
import time
from reward_space.utils.utils import kullback_leibler_divergence
from reward_space.inverse_reinforcement_learning.hessian_optimization import HeuristicOptimizerNegativeDefinite
from reward_space.proto_value_functions.proto_value_functions_estimator import DiscreteProtoValueFunctionsEstimator
from prettytable import PrettyTable

from reward_space.policy_gradient.policy_gradient_learner import \
    PolicyGradientLearner

def history_to_file(history, policy, pi_opt, J_opt, theta_opt):
    my_list = []
    for i in range(len(history)):
        theta, J, gradient = history[i]

        policy.set_parameter(theta)
        d_kl = kullback_leibler_divergence(pi_opt, policy.pi)

        mdp_wrap.set_policy(policy.pi2)
        J_true = mdp_wrap.compute_J()
        delta_J = J_opt - J
        delta_J_true = J_opt - J_true

        d_par = la.norm(theta - theta_opt)

        grad_norm = la.norm(gradient)

        my_list.append([d_kl, delta_J, delta_J_true, d_par, grad_norm])




def mdp_norm(f, mdp_wrap):
    d_sa_mu = mdp_wrap.compute_d_sa_mu()
    res = la.multi_dot([f, np.diag(d_sa_mu), f[:, np.newaxis]])
    return np.sqrt(res / sum(d_sa_mu))

def compute_trajectory_features(dataset, phi, state_space, action_space):
    i = 0
    j = 0
    nA = len(action_space)
    phi_tau = np.zeros((1, phi.shape[1]))
    while i < dataset.shape[0]:
        s_i = np.argwhere(state_space == dataset[i, 0])
        a_i = np.argwhere(action_space == dataset[i, 1])

        phi_tau[-1, :] += phi[np.asscalar(s_i * nA + a_i), :]

        if dataset[i, -1] == 1 and i < dataset.shape[0] - 1:
            phi_tau = np.vstack([phi_tau, np.zeros((1, phi.shape[1]))])
            j = j + 1
        i += 1
    return phi_tau


def plot_state_action_function(mdp, f, title, _cmap='coolwarm'):
    actions_names = ['S', 'N', 'W', 'E', 'up', 'down']
    Q = np.zeros((5,5,6))
    for i in range(0,f.shape[0],6):
        idx = list(mdp.decode(i/6))
        if idx[2] == 4 and idx[3] == 3:
            Q[idx[0], idx[1], :] = f[i:i+6]

    fig = plt.figure()

    x, y = np.meshgrid(range(1,6), range(1,6))
    for i in range(6):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        ax.plot_surface(x,y, Q[:,:, i], rstride=1, cstride=1, cmap=plt.get_cmap(_cmap),
                       linewidth=0.3, antialiased=True)
        ax.set_title(actions_names[i])
        ax.xaxis.set_ticks(np.arange(1, 6))
        ax.yaxis.set_ticks(np.arange(1, 6))

    fig.suptitle(title)

def plot_state_function(mdp, f, title, _cmap='coolwarm'):
    V = np.zeros((5,5))
    for i in range(0,f.shape[0]):
        idx = list(mdp.decode(i))
        if idx[2] == 4 and idx[3] == 3:
            V[idx[0], idx[1]] = f[i]

    fig = plt.figure()

    x, y = np.meshgrid(range(1,6), range(1,6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(x,y, V, rstride=1, cstride=1, cmap=plt.get_cmap(_cmap),
                       linewidth=0.3, antialiased=True)
    ax.xaxis.set_ticks(np.arange(1, 6))
    ax.yaxis.set_ticks(np.arange(1, 6))

    fig.suptitle(title)

def estimate_hessian(dataset, n_episodes, policy_gradient, policy_hessian, reward_features, state_space, action_space):

    if np.ndim(reward_features) == 1:
        reward_features = reward_features[:, np.newaxis]

    n_samples = dataset.shape[0]
    n_features = reward_features.shape[1]
    n_params = policy_hessian.shape[1]
    n_states, n_actions = len(state_space), len(action_space)

    episode_reward_features = np.zeros((n_episodes, n_features))
    episode_hessian = np.zeros((n_episodes, n_params, n_params))

    baseline = estimate_hessian_baseline(dataset, n_episodes, policy_gradient, policy_hessian, reward_features, state_space, action_space)

    i = 0
    episode = 0
    while i < n_samples:
        s = np.argwhere(state_space == dataset[i, 0])
        a = np.argwhere(action_space == dataset[i, 1])
        index = s * n_actions + a

        d = dataset[i, 4]

        episode_reward_features[episode, :] += d * reward_features[index, :].squeeze()
        episode_hessian[episode, :, :] += np.outer(
            policy_gradient[index, :].squeeze(), \
            policy_gradient[index, :].squeeze()) + \
            policy_hessian[index, :, :].squeeze()

        if dataset[i, -1] == 1:
            episode += 1

        i += 1

    episode_reward_features_baseline = episode_reward_features - baseline

    return_hessians = 1. / n_episodes * np.tensordot(episode_reward_features_baseline.T,
                                                     episode_hessian, axes=1)

    return return_hessians

def estimate_gradient(dataset, n_episodes, policy_gradient, reward_features, state_space, action_space):
    if np.ndim(reward_features) == 1:
        reward_features = reward_features[:, np.newaxis]

    n_samples = dataset.shape[0]
    n_features = reward_features.shape[1]
    n_params = policy_gradient.shape[1]
    n_states, n_actions = len(state_space), len(action_space)

    episode_reward_features = np.zeros((n_episodes, n_features))
    episode_gradient = np.zeros((n_episodes, n_params))

    baseline = estimate_gradient_baseline(dataset, n_episodes, policy_gradient, reward_features, state_space, action_space)

    i = 0
    episode = 0
    while i < n_samples:
        s = np.argwhere(state_space == dataset[i, 0])
        a = np.argwhere(action_space == dataset[i, 1])
        index = s * n_actions + a

        d = dataset[i, 4]

        episode_reward_features[episode, :] += d * reward_features[index, :].squeeze()
        episode_gradient[episode, :] +=  policy_gradient[index, :].squeeze()

        if dataset[i, -1] == 1:
            episode += 1

        i += 1

    episode_reward_features_baseline = episode_reward_features - baseline

    return_gradient = 1. / n_episodes * np.dot(episode_reward_features_baseline.T,
                                                     episode_gradient)

    return return_gradient

def estimate_gradient_baseline(dataset, n_episodes, policy_gradient, reward_features, state_space, action_space):
    n_samples = dataset.shape[0]
    n_features = reward_features.shape[1]
    n_params = policy_gradient.shape[1]
    n_states, n_actions = len(state_space), len(action_space)

    episode_reward_features = np.zeros((n_episodes, n_features))
    episode_gradient = np.zeros((n_episodes, n_params))
    numerator = denominator = 0.

    i = 0
    episode = 0
    while i < n_samples:
        s = np.argwhere(state_space == dataset[i, 0])
        a = np.argwhere(action_space == dataset[i, 1])
        index = s * n_actions + a

        d = dataset[i, 4]

        episode_reward_features[episode, :] += d * reward_features[index, :].squeeze()
        episode_gradient[episode, :] += policy_gradient[index, :].squeeze()

        if dataset[i, -1] == 1:
            vectorized_gradient = episode_gradient[episode, :].ravel()
            numerator += episode_reward_features[episode, :] * la.norm(vectorized_gradient) ** 2
            denominator += la.norm(vectorized_gradient) ** 2
            episode += 1

        i += 1

    baseline = numerator / denominator

    return baseline

def estimate_hessian_baseline(dataset, n_episodes, policy_gradient, policy_hessian, reward_features, state_space, action_space):
    n_samples = dataset.shape[0]
    n_features = reward_features.shape[1]
    n_params = policy_hessian.shape[1]
    n_states, n_actions = len(state_space), len(action_space)

    episode_reward_features = np.zeros((n_episodes, n_features))
    episode_hessian = np.zeros((n_episodes, n_params, n_params))
    numerator = denominator = 0.

    i = 0
    episode = 0
    while i < n_samples:
        s = np.argwhere(state_space == dataset[i, 0])
        a = np.argwhere(action_space == dataset[i, 1])
        index = s * n_actions + a

        d = dataset[i, 4]

        episode_reward_features[episode, :] += d * reward_features[index, :].squeeze()
        episode_hessian[episode, :, :] += np.outer(
            policy_gradient[index, :].squeeze(), \
            policy_gradient[index, :].squeeze()) + policy_hessian[index, :,
                                                   :].squeeze()

        if dataset[i, -1] == 1:
            vectorized_hessian = episode_hessian[episode, :, :].ravel()
            numerator += episode_reward_features[episode, :] * la.norm(vectorized_hessian) ** 2
            denominator += la.norm(vectorized_hessian) ** 2
            episode += 1

        i += 1

    baseline = numerator / denominator

    return baseline

def trace_minimization(hessians, features, threshold):
    n_states_actions = hessians.shape[0]
    n_parameters = hessians.shape[1]
    w = cvxpy.Variable(n_states_actions)
    final_hessian = hessians[0] * w[0]
    for i in range(1, n_states_actions):
        final_hessian += hessians[i] * w[i]

    objective = cvxpy.Minimize(cvxpy.trace(final_hessian))
    constraints = [final_hessian + threshold * np.eye(n_parameters) << 0,
                   cvxpy.sum_entries(w) == 1]
    problem = cvxpy.Problem(objective, constraints)

    result = problem.solve(verbose=True)
    return w.value, final_hessian.value, result


def maximum_eigenvalue_minimizarion(hessians, features, threshold):
    n_states_actions = hessians.shape[0]
    n_parameters = hessians.shape[1]
    w = cvxpy.Variable(n_states_actions)
    final_hessian = hessians[0] * w[0]
    for i in range(1, n_states_actions):
        final_hessian += hessians[i] * w[i]

    objective = cvxpy.Minimize(cvxpy.lambda_max(final_hessian))
    constraints = [final_hessian + threshold * np.eye(n_parameters) << 0,
                   cvxpy.norm(w) <= 1]
    problem = cvxpy.Problem(objective, constraints)

    result = problem.solve(verbose=True)
    return w.value, final_hessian.value, result

def maximum_entropy(dataset, n_episodes, policy_gradient, reward_features, state_space, action_space):

    n_samples = dataset.shape[0]
    n_features = reward_features.shape[1]
    n_params = policy_gradient.shape[1]
    n_states, n_actions = len(state_space), len(action_space)

    episode_reward_features = np.zeros((n_episodes, n_features))
    episode_hessian = np.zeros((n_episodes, n_params, n_params))

    i = 0
    episode = 0
    while i < n_samples:
        s = np.argwhere(state_space == dataset[i, 0])
        a = np.argwhere(action_space == dataset[i, 1])
        index = s * n_actions + a

        d = dataset[i, 4]

        episode_reward_features[episode, :] += d * reward_features[index, :].squeeze()
        if dataset[i, -1] == 1:
            episode += 1

        i += 1

    import scipy.optimize as opt

    def loss(x):
        episode_reward = np.dot(episode_reward_features, x)
        exp_episode_reward = np.exp(episode_reward)
        partition_function = np.sum(exp_episode_reward)
        episode_prob = exp_episode_reward / partition_function
        log_episode_prob = np.log(episode_prob)
        return -np.sum(log_episode_prob)

    def constraint(x):
        return la.norm(np.dot(reward_features, x)) - 1

    res = opt.minimize(loss, np.ones(n_features),
                       constraints=({'type': 'eq', 'fun': constraint}),
                       options={'disp': True},
                       tol=1e-24)
    print(res)

    return res.x


def build_state_features(mdp, binary=True):
    '''
    Builds the features of the state based on the current position of the taxi,
    whether it has already pick up the passenger, the relative position of the
    destination and source cell.

    :param mdp: the taxi mdp
    :param binary: whether the feature has to be one hot encoded
    :return: a (n_states, n_features) matrix containing the features for each
             state
    '''
    state_features = np.zeros((mdp.nS, 6))
    locs = [(0, 0), (0, 4), (4, 0), (4, 3)] #source and destination location
    for i in range(mdp.nS):
        lst = list(mdp.decode(i))
        pos_x, pos_y = lst[0], lst[1]

        if lst[2] == 4:
            start_x, start_y = 5, 5
        else:
            start_x, start_y = locs[lst[2]]
        arr_x, arr_y = locs[lst[3]]

        if start_x == 5:
            if arr_x == pos_x:
                delta_x = 2
            else:
                delta_x = 1 if arr_x > pos_x else 0

            if arr_y == pos_y:
                delta_y = 2
            else:
                delta_y = 1 if arr_y > pos_y else 0
        else:
            if start_x == pos_x:
                delta_x = 2
            else:
                delta_x = 1 if start_x > pos_x else 0

            if start_y == pos_y:
                delta_y = 2
            else:
                delta_y = 1 if start_y > pos_y else 0

        if delta_y == 2 and delta_x == 2:
            if start_x == 5:
                is_pos = 0
            else:
                is_pos = 1
        else:
            is_pos = 2


        '''
        A feature is encoded as a 5-tuple:
        - the (x,y) position of the taxi combined as 5x+y
        - the index of the starting location
        - the index of the destination location
        - whether the current x-position is on the same row, on the left or on the
          right wrt the source position if the passenger has not been pick up yet
          or wrt the destination position
        - whether the current y-position is on the same column, on the left or on the
          right wrt the source position if the passenger has not been pick up yet
          or wrt the destination position
        '''

        state_features[i, :] = [pos_x*5 + pos_y, lst[2], lst[3], delta_x, delta_y, is_pos]

    if not binary:
        return state_features
    else:
        enc = OneHotEncoder(n_values=[25, 4, 4, 2, 2, 3], sparse=False,
                            handle_unknown='ignore')
        enc.fit(state_features)
        state_features_binary = enc.transform(state_features)
        state_features_binary = state_features_binary[:, :state_features_binary.shape[1]-1]
        return state_features_binary


def build_state_features2(mdp, binary=True):
    '''
    Builds the features of the state based on the current position of the taxi,
    whether it has already pick up the passenger, the relative position of the
    destination and source cell.

    :param mdp: the taxi mdp
    :param binary: whether the feature has to be one hot encoded
    :return: a (n_states, n_features) matrix containing the features for each
             state
    '''
    state_features = np.zeros((mdp.nS, 5))
    locs = [(0, 0), (0, 4), (4, 0), (4, 3)] #source and destination location
    for i in range(mdp.nS):
        lst = list(mdp.decode(i))
        pos_x, pos_y = lst[0], lst[1]

        if lst[2] == 4:
            start_x, start_y = 5, 5
        else:
            start_x, start_y = locs[lst[2]]
        arr_x, arr_y = locs[lst[3]]

        if start_x == 5:
            if arr_x == pos_x:
                delta_x = 0
            else:
                delta_x = 2 if arr_x > pos_x else 1

            if arr_y == pos_y:
                delta_y = 0
            else:
                delta_y = 2 if arr_y > pos_y else 1
        else:
            if start_x == pos_x:
                delta_x = 0
            else:
                delta_x = 2 if start_x > pos_x else 1

            if start_y == pos_y:
                delta_y = 0
            else:
                delta_y = 2 if start_y > pos_y else 1

        '''
        A feature is encoded as a 5-tuple:
        - the (x,y) position of the taxi combined as 5x+y
        - the index of the starting location
        - the index of the destination location
        - whether the current x-position is on the same row, on the left or on the
          right wrt the source position if the passenger has not been pick up yet
          or wrt the destination position
        - whether the current y-position is on the same column, on the left or on the
          right wrt the source position if the passenger has not been pick up yet
          or wrt the destination position
        '''

        state_features[i, :] = [pos_x*5 + pos_y, lst[2], lst[3], delta_x, delta_y]

    if not binary:
        return state_features
    else:
        enc = OneHotEncoder(n_values=[25, 4, 4, 3, 3], sparse=False,
                            handle_unknown='ignore')
        enc.fit(state_features)
        state_features_binary = enc.transform(state_features)
        return state_features_binary

def fit_maximum_likelihood_policy(state_features, optimal_action):
    '''
    Finds the maximum likelihood Boltzmann policy that best approximates a given
    deterministic policy.
    :param state_features: a (n_states, n_features) matrix representing the
                           feature vector for each state
    :param optimal_action: a (n_states,) vector containg for each state the index
                           of the optimal action
    :return: a pair: (action_weights, pi_prox)
             action_weights is a (n_actions, n_features) matrix representing the
                parameter estimated for each action
             pi_prox is a (n_states,n_actions) matrix representing the approximated
                probability distribution
    '''

    lr = LogisticRegression(penalty='l2',
                            tol=1e-16,
                            C=np.inf,
                            solver='newton-cg',
                            fit_intercept=False,
                            intercept_scaling=1,
                            max_iter=300,
                            multi_class='multinomial',
                            verbose=0,
                            n_jobs=1)

    lr.fit(state_features, optimal_action)
    action_weights = lr.coef_
    pi_prox = lr.predict_proba(state_features)
    return action_weights, pi_prox


def estimate(features, target, scorer, ks):
    error = []
    target_hat = []
    for k in ks:
        X = la2.range(features[:, :k])
        y_hat, _, _, _, = la2.lsq(X, target)
        error.append(scorer.score(target, y_hat))
        target_hat.append(y_hat)
    return target_hat, error

class Scorer(object):

    def __init__(self,
                 weights,
                 p=2):
        self.weights = weights
        self.p = p

    def score(self,
              actual,
              predicted):

        return self.norm(actual - predicted) / self.norm(actual)

    def norm(self, vector):
        weighted_vector = vector * np.power(self.weights, 1. / self.p)
        return la.norm(weighted_vector, self.p)

    def __str__(self):
        return '||.||%s weighted' % self.p


class RangeScaler(object):

    def __init__(self, copy=True):
        self.copy = copy

    def fit_transform(self, X):
        if self.copy:
            self.X = np.copy(X)
        else:
            self.X = X

        _max, _min = self.X.max(axis=0), self.X.min(axis=0)
        self.X = self.X / (_max - _min)
        return self.X

if __name__ == '__main__':

    mytime = time.time()

    plot = False
    plot_gradient = False
    perform_estimation_pvf = True
    plot_pvf = True
    perform_estimation_gpvf = True
    plot_gpvf = True
    kmin, kmax, kstep = 0, 501, 50
    on_policy = True
    off_policy = False
    methods = []
    if on_policy:
        methods.append('on-policy')
    if off_policy:
        methods.append('off-policy')
    plot_hessians = True

    #k_pvf = [10, 20, 50, 100, 200]
    k_pvf = [10, 100]

    tol = 1e-24
    mdp = TaxiEnv()
    mdp.horizon = 100
    n_episodes = 1000

    print('Computing optimal policy...')
    opt_policy = TaxiEnvPolicy()
    pi_opt = opt_policy.pi

    print('Building state features...')
    state_features = build_state_features(mdp, binary=True)

    print('Computing maximum likelihood Boltrzman policy...')
    action_weights, pi_prox = fit_maximum_likelihood_policy(state_features, opt_policy.policy.values())
    d_kl = np.sum(pi_opt * np.log(pi_opt / pi_prox + 1e-24))
    n_parameters = action_weights.shape[0] * action_weights.shape[1]
    print('Number of features %s Number of parameters %s' % (state_features.shape[1], n_parameters))
    print('KL divergence = %s' % d_kl)

    policy =  BoltzmannPolicy(state_features, action_weights)
    theta_opt = np.copy(policy.state_action_parameters)

    print('Collecting samples from optimal approx policy...')
    dataset = evaluation.collect_episodes(mdp, policy, n_episodes)
    n_samples = dataset.shape[0]
    print('Dataset made of %s samples' % n_samples)

    mdp_wrap = DiscreteMdpWrapper(mdp, episodic=True)
    pi_opt = opt_policy.get_distribution()
    pi = policy.get_pi()
    G = policy.gradient_log()

    # Optimal deterministic policy
    mdp_wrap.set_policy(pi_opt)
    J_opt = mdp_wrap.compute_J()

    # Optimal approx policy
    mdp_wrap.set_policy(pi)
    d_sa_mu = mdp_wrap.compute_d_sa_mu()
    D = np.diag(d_sa_mu)
    R = mdp_wrap.non_ep_R
    J_true = mdp_wrap.compute_J()
    Q_true = mdp_wrap.compute_Q_function()
    pi_tilde = np.repeat(pi, mdp_wrap.nA, axis=0)
    A_true = (np.eye(mdp_wrap.nA * mdp_wrap.nS) - pi_tilde).dot(Q_true)
    V_true = mdp_wrap.compute_V_function()[:mdp_wrap.nS]

    if plot:
        plot_state_action_function(mdp, Q_true, 'Q-function')
        plot_state_action_function(mdp, R , 'R-function')
        plot_state_action_function(mdp, A_true, 'A-function')
        plot_state_action_function(mdp, d_sa_mu, 'd(s,a)')
        plot_state_function(mdp, V_true, 'V-function')

    from reward_space.inverse_reinforcement_learning.linear_programming_apprenticeship_learning import \
        LinearProgrammingApprenticeshipLearning

    #---------------------------------------------------------------------------
    #Sample estimations of return and gradient
    print('-' * 100)

    print('Estimating return and gradient...')
    estimator = DiscreteEnvSampleEstimator(dataset,
                                           mdp_wrap.gamma,
                                           mdp_wrap.state_space,
                                           mdp_wrap.action_space)

    d_s_mu_hat = estimator.get_d_s_mu()
    d_sa_mu_hat = np.dot(pi.T, d_s_mu_hat)
    D_hat = np.diag(d_sa_mu_hat)
    J_hat = estimator.get_J()

    print('Expected reward opt det policy J_opt = %g' % J_opt)
    print('True expected reward approx opt policy J_true = %g' % J_true)
    print('Estimated expected reward approx opt policy J_hat = %g' % J_hat)

    grad_J_hat = la.multi_dot([G.T, D_hat, Q_true])
    grad_J_true = la.multi_dot([G.T, D, Q_true])
    print('Dimension of the subspace %s/%s' % (la.matrix_rank(np.dot(G.T, D)), n_parameters))
    print('True policy gradient (2-norm) DJ_true = %s' % la.norm(grad_J_true, 2))
    print('Estimated policy gradient (2-norm) DJ_hat = %s' % la.norm(grad_J_hat, 2))

    if plot_gradient:
        fig, ax = plt.subplots()
        ax.set_xlabel('Index')
        ax.set_ylabel('Singular value')
        fig.suptitle('Gradient singular values')
        s = la.svd(G.T, compute_uv=False)
        plt.plot(np.arange(len(s)), s, marker='o', label='Gradient * d(s,a)')

    print('-' * 100)

    print('Computing Q-function approx space...')
    X = np.dot(G.T, D_hat)
    phi = la2.nullspace(X)

    print('Computing reward function approx space...')
    Y = np.dot(np.eye(mdp_wrap.nA * mdp_wrap.nS) - pi_tilde, phi)
    psi = la2.range(Y)

    del X
    del Y
    #---------------------------------------------------------------------------
    #Hessian estimation

    print('Estimating hessians...')

    names = []
    basis_functions = []
    gradients = []
    hessians = []

    H = policy.hessian_log()

    hessian_hat = estimate_hessian(dataset, n_episodes,G, H, psi, \
                                mdp_wrap.state_space, mdp_wrap.action_space)

    eigval_hat, _ = la.eigh(hessian_hat)
    eigmax_hat, eigmin_hat = eigval_hat[:, -1], eigval_hat[:, 0]
    trace_hat = np.trace(hessian_hat, axis1=1, axis2=2)

    minmax_prod = eigmax_hat * eigmin_hat
    pos_idx, neg_idx, ind_idx = np.argwhere(np.bitwise_and(minmax_prod >= 0, eigmax_hat >=0)),\
                                np.argwhere(np.bitwise_and(minmax_prod >= 0, eigmax_hat < 0)),\
                                np.argwhere(minmax_prod < 0)

    neg_idx = np.concatenate([neg_idx, np.argwhere(np.bitwise_and(minmax_prod < 0, eigmax_hat < 1e-12))])

    psi[:, pos_idx] *= -1
    psi = psi[:, np.concatenate([pos_idx, neg_idx]).ravel()]
    hessian_hat_neg = np.copy(hessian_hat)
    hessian_hat_neg[pos_idx] *= -1
    hessian_hat_neg = hessian_hat_neg[np.concatenate([pos_idx, neg_idx]).ravel()]

    names.append('Reward function')
    basis_functions.append(R / la.norm(R))
    names.append('Advantage function')
    basis_functions.append(A_true / la.norm(A_true))

    '''
    HEURISTIC SOLUTION
    max eig minimization or hessian heuristic in this case are
    the same since the dimension of parameter space is 1
    '''

    optimizer = HeuristicOptimizerNegativeDefinite(hessian_hat_neg)
    w = optimizer.fit(skip_check=False)
    grbf = np.dot(psi, w)
    names.append('GRBF heuristic solution minimizer')
    basis_functions.append(grbf)


    '''
    MAXIMUM ENTROPY IRL

    '''
    '''
    print(time.time())
    print('-' * 100)
    print('Estimating maximum entropy reward...')
    from reward_space.inverse_reinforcement_learning.maximum_entropy_irl import MaximumEntropyIRL
    me = MaximumEntropyIRL(dataset)
    w = me.fit(G, psi, mdp_wrap.state_space, mdp_wrap.action_space)
    w = w / la.norm(w)

    names.append('Maximum entropy GRBF')
    basis_functions.append(np.dot(psi, w))
    print(time.time())
    '''

    '''
    PROTO-VALUE FUNCTIONS
    '''

    print('Estimating proto value functions...')
    dpvf = DiscreteProtoValueFunctionsEstimator(mdp_wrap.state_space,
                                               mdp_wrap.action_space)
    dpvf.fit(dataset)

    for k in k_pvf:
        _, phi_pvf = dpvf.transform(k)
        phi_pvf = phi_pvf.sum(axis=1).ravel()
        pvf_norm = phi_pvf / la.norm(phi_pvf)

        names.append('PVF %s' % k)
        basis_functions.append(pvf_norm)

    lpal = LinearProgrammingApprenticeshipLearning(mdp_wrap.non_ep_P,
                                                   opt_policy.PI,
                                                   mdp_wrap.nA,
                                                   mdp_wrap.gamma, 1, 0.)
    r_s_lpal = lpal.fit()
    r_sa_lpal = np.repeat(r_s_lpal, mdp_wrap.nA)
    names.append('LPAL reward')
    basis_functions.append(r_sa_lpal)

    '''
    Gradient and hessian estimation
    '''

    # Rescale rewards into to have difference between max and min equal to 1
    scaler = RangeScaler()

    scaled_grbf = scaler.fit_transform(grbf[:, np.newaxis]).ravel()

    scaled_basis_functions = []
    for bf in basis_functions:
        sbf = scaler.fit_transform(bf[:, np.newaxis]).ravel()
        scaled_basis_functions.append(sbf)

    gradients, hessians, gradient_norms2, gradient_normsinf, eigvals, traces, eigvals_max = [], [], [], [], [], [], []
    print('Estimating gradient and hessians...')
    for sbf in scaled_basis_functions:
        gradient = estimate_gradient(dataset, n_episodes,G, sbf, \
                                mdp_wrap.state_space, mdp_wrap.action_space)[0]
        hessian = estimate_hessian(dataset, n_episodes, G, H, sbf, \
                              mdp_wrap.state_space, mdp_wrap.action_space)[0]
        gradients.append(gradient)
        hessians.append(hessian)
        eigval, _ = la.eigh(hessian)
        gradient_norms2.append(la.norm(gradient, 2))
        gradient_normsinf.append(la.norm(gradient, np.inf))
        eigvals_max.append(eigval[-1])
        eigvals.append(eigval)
        traces.append(np.trace(hessian))


    t = PrettyTable()
    t.add_column('Basis function', names)
    t.add_column('Gradient norm 2', gradient_norms2)
    t.add_column('Gradient norm inf', gradient_normsinf)
    t.add_column('Hessian  max eigenvalue', eigvals_max)
    t.add_column('Hessian trace', traces)
    print(t)

    print('Saving results...')
    gradients_np = np.array(gradients)
    hessians_np = np.array(hessians)
    gradient_norms2_np = np.array(gradient_norms2)
    gradient_normsinf_np = np.array(gradient_normsinf)
    eigvals_max_np = np.array(eigvals_max)
    eigvals_np = np.array(eigvals)
    traces_np = np.array(traces)

    saveme = np.zeros(8, dtype=object)
    saveme[0] = names
    saveme[1] = gradients_np
    saveme[2] = hessians_np
    saveme[3] = gradient_norms2_np
    saveme[4] = gradient_normsinf_np
    saveme[5] = eigvals_max_np
    saveme[6] = eigvals_np
    saveme[7] = traces_np
    np.save('data/taxi_gradients_hessians_%s' % mytime, saveme)

    if plot:
        fig, ax = plt.subplots()
        ax.set_xlabel('trace')
        ax.set_ylabel('max eigval')
        fig.suptitle('Hessians')
        ax.scatter(trace_hat, eigmax_hat, color='k', marker='+',
                   label='Estimated hessians')
        for i in range(len(names)):
            ax.plot(traces_np[i], eigvals_max_np[i], marker='+', label=names[i])
        ax.legend(loc='upper right')

        _range = np.arange(len(eigvals_np[0]))
        fig, ax = plt.subplots()
        ax.set_xlabel('index')
        ax.set_ylabel('eigenvalue')
        fig.suptitle('Hessian Eigenvalues')
        for i in range(len(names)):
            ax.plot(_range, eigvals_np[i], marker='+', label=names[i])

        ax.legend(loc='upper right')
        plt.yscale('symlog', linthreshy=1e-12)

    '''
    REINFORCE training
    '''

    print('-' * 100)
    print('Estimating d(s,a)...')

    count_sa_hat = estimator.get_count_sa()
    count_sa_hat /= count_sa_hat.max()

    learner = PolicyGradientLearner(mdp, policy, lrate=0.02, verbose=1,
                                    max_iter_opt=200, tol_opt=-1., tol_eval=0.,
                                    estimator='reinforce',
                                    gradient_updater='adam')

    theta0 = np.zeros((n_parameters, 1))

    reward = scaled_basis_functions[0]
    theta, history_r = learner.optimize(theta0, reward=lambda traj: reward[
        map(int, traj[:, 0] * 6 + traj[:, 1])], return_history=True)

    advantage = scaled_basis_functions[1]
    theta, history_a = learner.optimize(theta0, reward=lambda traj: advantage[
        map(int, traj[:, 0] * 6 + traj[:, 1])], return_history=True)

    labels = ['Reward function', 'Advantage function']
    histories = [history_r, history_a]

    for i in range(2, len(scaled_basis_functions)):
        print(names[i])
        sbf = scaled_basis_functions[i]
        penalized_sbf = np.copy(sbf)
        penalized_sbf[count_sa_hat == 0] = min(penalized_sbf)
        theta, history = learner.optimize(theta0, reward=lambda traj: penalized_sbf[
            map(int, traj[:, 0] * 6 + traj[:, 1])],return_history=True)
        histories.append(history)
    labels = labels + map(lambda x: x + 'penalized', names[2:])

    histories = np.array(histories)

    t = PrettyTable()
    t.add_column('Basis function', labels)
    #t.add_column('Final parameter', la.norm(histories[:, -1, 0]))
    t.add_column('Final return', histories[:, -1, 1])
    #t.add_column('Final gradient', la.norm(histories[:, -1, 2]))
    print(t)

    if plot:
        _range = np.arange(201)
        fig, ax = plt.subplots()
        ax.set_xlabel('average return')
        ax.set_ylabel('iterations')
        fig.suptitle('REINFORCE - Average return')

        ax.plot([0, 100], [estimator.J, estimator.J], color='k', label='Optimal return')
        for i in range(6):
            ax.plot(_range, histories[i, :, 1], marker='+', label=labels[i])

        ax.legend(loc='upper right')

    histories = np.array(histories)
    saveme = np.zeros(2, dtype=object)
    saveme[0] = labels
    saveme[1] = histories
    np.save('data/taxi_comparision_%s' % mytime, saveme)

    '''


    hessian_true = ml_estimator.estimate_hessian(G, H, r_true, \
                        True, mdp_wrap.state_space, mdp_wrap.action_space)[0]

    hessian_true_a = ml_estimator.estimate_hessian(G, H, a_true, \
                                True, mdp_wrap.state_space, mdp_wrap.action_space)[0]

    '''

    '''
    hessian_true = estimate_hessian(dataset, n_episodes, G, H, r_true, \
                        mdp_wrap.state_space, mdp_wrap.action_space)[0]

    hessian_true_a = estimate_hessian(dataset, n_episodes,G, H, a_true, \
                                mdp_wrap.state_space, mdp_wrap.action_space)[0]

    hessian_hat = estimate_hessian(dataset, n_episodes,G, H, psi, \
                                mdp_wrap.state_space, mdp_wrap.action_space)
    '''
    '''
    print('Computing traces...')
    trace_true = np.trace(hessian_true)
    trace_true_a = np.trace(hessian_true_a)
    trace_hat = np.trace(hessian_hat, axis1=1, axis2=2)

    min_trace_idx = trace_hat.argmin()
    max_trace_idx = trace_hat.argmax()

    print('Computing max eigenvalue...')
    eigval_true, _ = la.eigh(hessian_true)
    eigmax_true = eigval_true[-1]

    eigval_true_a, _ = la.eigh(hessian_true_a)
    eigmax_true_a = eigval_true_a[-1]

    eigval_hat, _ = la.eigh(hessian_hat)
    eigmax_hat = eigval_hat[:, -1]

    semidef_idx = eigmax_hat < 1e-13

    from reward_space.inverse_reinforcement_learning.hessian_optimization import HeuristicOptimizerNegativeDefinite
    ho = HeuristicOptimizerNegativeDefinite(hessian_hat[semidef_idx])
    w = ho.fit()
    best_hessian = np.tensordot(w, hessian_hat[semidef_idx], axes=1)

    trace_best = np.trace(best_hessian)
    eigval_best, _ = la.eigh(best_hessian)
    eigmax_best = eigval_best[-1]

    del hessian_hat
    del H

    if plot_hessians:
        fig, ax = plt.subplots()
        ax.set_xlabel('index')
        ax.set_ylabel('eigenvalue')
        fig.suptitle('Hessian Eigenvalues')
        ax.plot(np.arange(len(eigval_true)), eigval_true, color='r', marker='+',
                   label='Reward function')
        ax.plot(np.arange(len(eigval_true)), eigval_true_a, color='g', marker='+',
                   label='Advantage function')
        ax.plot(np.arange(len(eigval_true)), eigval_hat[min_trace_idx], color='b', marker='+',
                   label='Feature with smallest trace')
        ax.plot(np.arange(len(eigval_true)), eigval_hat[max_trace_idx], color='m', marker='+',
                   label='Feature with largest trace')
        ax.plot(np.arange(len(eigval_true)), eigval_best, color='y', marker='+',
                label='Best')
        ax.plot(np.arange(len(eigval_me)), eigval_me, color='k', marker='+',
                label='Maximum entropy')
        ax.plot(np.arange(len(eigval_25)), eigval_25, color='b', marker='o',
                label='PVF 25')
        ax.plot(np.arange(len(eigval_50)), eigval_50, color='r', marker='o',
                label='PVF 50')

        ax.legend(loc='upper right')
        plt.yscale('symlog', linthreshy=1e-12)


    best = np.dot(psi[:,semidef_idx], w)
    min_value = min(best)
    best_action_per_state = d_sa_mu_hat.reshape(mdp_wrap.nS, mdp_wrap.nA).argmax(axis=1)

    best2 = min_value * np.ones(3000)
    best2[np.arange(500)*6 + best_action_per_state] = best[np.arange(500)*6 + best_action_per_state]

    r_norm = R / (max(R) - min(R))
    a_norm = A_true / (max(A_true) - min(A_true))
    best_norm = best2 / (max(best2) - min(best2))

    from policy_gradient.policy_gradient_learner import PolicyGradientLearner
    learner = PolicyGradientLearner(mdp, policy, lrate=0.05, verbose=1,
                                    max_iter_opt=100, tol_opt=0., tol_eval=0.,
                                    estimator='reinforce')
    theta0 = np.zeros((n_parameters, 1))

    best_norm = best_norm.ravel()
    policy.set_parameter(theta0)
    theta = learner.optimize(theta0, reward=lambda traj: a_norm[ map(int, traj[:, 0] * 6 + traj[:, 1])])
    policy.set_parameter(theta)
    print('Collecting samples from optimal approx policy...')
    dataset = evaluation.collect_episodes(mdp, policy, n_episodes)
    n_samples = dataset.shape[0]
    print('Dataset made of %s samples' % n_samples)


    from reward_space.inverse_reinforcement_learning.maximum_entropy_irl import MaximumEntropyIRL
    me = MaximumEntropyIRL(dataset)
    w = me.fit(G, psi, mdp_wrap.state_space, mdp_wrap.action_space)
    best_me = np.dot(psi, w)

    best_me_norm = best_me / (max(best_me) - min(best_me))
    best_me_norm = best_me_norm.ravel()

    min_value = min(best_me_norm)
    best2 = min_value * np.ones(3000)
    best2[np.arange(500)*6 + best_action_per_state] = best_me_norm[np.arange(500)*6 + best_action_per_state]
    best_me = best2

    hessian_me = estimate_hessian(dataset, n_episodes,G, H, best_me[:, np.newaxis] / la.norm(best_me), \
                                mdp_wrap.state_space, mdp_wrap.action_space)[0]

    trace_me = np.trace(hessian_me)
    eigval_me, _ = la.eigh(hessian_me)
    eigmax_me = eigval_me.max()

    learner = PolicyGradientLearner(mdp, policy, lrate=0.05, verbose=1,
                                    max_iter_opt=100, tol_opt=0., tol_eval=0.,
                                    estimator='reinforce')
    theta0 = np.zeros((n_parameters, 1))

    best_norm = best_norm.ravel()
    policy.set_parameter(theta0)
    theta = learner.optimize(theta0, reward=lambda traj: best2[ map(int, traj[:, 0] * 6 + traj[:, 1])])
    policy.set_parameter(theta)
    print('Collecting samples from optimal approx policy...')
    dataset = evaluation.collect_episodes(mdp, policy, n_episodes)
    n_samples = dataset.shape[0]
    print('Dataset made of %s samples' % n_samples)

    #---------------------------------------------------------------------------
    #Proto-value functions
    from reward_space.proto_value_functions.proto_value_functions_estimator import DiscreteProtoValueFunctionsEstimator
    pvf_model = DiscreteProtoValueFunctionsEstimator(mdp_wrap.state_space,
                                               mdp_wrap.action_space)

    pvf_model.fit(dataset)
    _, pvf25 = pvf_model.transform(25)
    pvf25 = pvf25.sum(axis=1)
    pvf25 /= la.norm(pvf25)
    _, pvf50 = pvf_model.transform(50)
    pvf50 = pvf50.sum(axis=1)
    pvf50 /= la.norm(pvf50)

    hessian_pvf25 = estimate_hessian(dataset, n_episodes,G, H, pvf25[:, np.newaxis], \
                                mdp_wrap.state_space, mdp_wrap.action_space)[0]

    hessian_pvf50 = estimate_hessian(dataset, n_episodes,G, H, pvf50[:, np.newaxis], \
                                mdp_wrap.state_space, mdp_wrap.action_space)[0]

    trace25 = np.trace(hessian_pvf25)
    trace50 = np.trace(hessian_pvf50)

    eigval_25, _ = la.eigh(hessian_pvf25)
    eigval_50, _ = la.eigh(hessian_pvf50)

    best_me_norm = best_me / (max(best_me) - min(best_me))
    pvf25_norm = pvf25 / (max(pvf25) - min(pvf25))
    pvf50_norm = pvf25 / (max(pvf50) - min(pvf50))

    #---------------------------------------------------------------------------
    #Leanring
    from reward_space.policy_gradient.policy_gradient_learner import PolicyGradientLearner
    learner = PolicyGradientLearner(mdp, policy, lrate=0.2, verbose=1,
                                    max_iter_opt=200, tol_opt=-1, tol_eval=0.,
                                    estimator='reinforce')
    theta0 = np.zeros((n_parameters, 1))

    theta, history_r = learner.optimize(theta0, reward=lambda traj: r_norm[ map(int, traj[:, 0] * 6 + traj[:, 1])], return_history=True)

    theta, history_a = learner.optimize(theta0, reward=lambda traj: a_norm[ map(int, traj[:, 0] * 6 + traj[:, 1])], return_history=True)

    theta, history_b = learner.optimize(theta0, reward=lambda traj: best_norm[
        map(int, traj[:, 0] * 6 + traj[:, 1])], return_history=True)

    theta, history_me = learner.optimize(theta0, reward=lambda traj: best_me_norm[
        map(int, traj[:, 0] * 6 + traj[:, 1])], return_history=True)

    theta, history_pvf25 = learner.optimize(theta0, reward=lambda traj: pvf25_norm[
        map(int, traj[:, 0] * 6 + traj[:, 1])], return_history=True)

    theta, history_pvf50 = learner.optimize(theta0, reward=lambda traj: pvf50_norm[
        map(int, traj[:, 0] * 6 + traj[:, 1])], return_history=True)

    np.save('r', np.array(history_r))
    np.save('a', np.array(history_a))
    np.save('b', np.array(history_b))
    np.save('me', np.array(history_me))
    np.save('pvf25', np.array(history_pvf25))
    np.save('pvf50', np.array(history_pvf50))

    fig, ax = plt.subplots()
    ax.set_xlabel('iterations')
    ax.set_ylabel('average reward')
    fig.suptitle('REINFORCE')
    ax.plot(np.arange(201), np.array(history_r)[:,1], color='r', marker='+',
            label='Reward function')
    ax.plot(np.arange(201), np.array(history_a)[:,1], color='g', marker='+',
            label='Advantage function')
    ax.plot(np.arange(201), np.array(history_b)[:,1], color='y', marker='+',
            label='Best')
    ax.plot(np.arange(201), np.array(history_me)[:,1], color='k', marker='+',
            label='Maximum entropy')
    ax.plot(np.arange(201), np.array(history_pvf25)[:,1], color='b', marker='o',
            label='PVF 25')
    ax.plot(np.arange(201), np.array(history_pvf50)[:,1], color='r', marker='o',
            label='PVF 50')

    ax.legend(loc='lower right')


    print('Collecting samples from optimal approx policy...')
    dataset = evaluation.collect_episodes(mdp, policy, n_episodes)
    n_samples = dataset.shape[0]
    print('Dataset made of %s samples' % n_samples)


        learner = PolicyGradientLearner(mdp, policy, lrate=0.05, verbose=1,
                                        max_iter_opt=10, tol_opt=0., tol_eval=0.,
                                        estimator='reinforce')
        theta0 = np.zeros((n_parameters, 1))

        best_norm = best_norm.ravel()
        policy.set_parameter(theta0)
        theta = learner.optimize(theta0, reward=lambda traj: bf_norm[ map(int, traj[:, 0] * 6 + traj[:, 1])])
        policy.set_parameter(theta)
        print('Collecting samples from optimal approx policy...')
        dataset = evaluation.collect_episodes(mdp, policy, n_episodes)
        n_samples = dataset.shape[0]
        print('Dataset made of %s samples' % n_samples)

    if plot_hessians:
        fig, ax = plt.subplots()
        ax.set_xlabel('trace')
        ax.set_ylabel('max eigval')
        fig.suptitle('Hessians')
        ax.scatter(trace_hat, eigmax_hat, color='b', marker='+',
                   label='Estimated hessians')
        ax.scatter(trace_true, eigmax_true, color='r', marker='o', label='Reward function')
        ax.scatter(trace_true_a, eigmax_true_a, color='g', marker='o', label='Advantage function')
        ax.scatter(trace_best, eigmax_best, color='y', marker='o',
                   label='Best')
        ax.legend(loc='upper right')
        # ax.scatter(trace_trace, eigmax_trace, color='g', marker='d', s=100)
        # ax.scatter(trace_eigval, eigmax_eigval, color='y', marker='d', s=100)

    ind = trace_hat[eigmax_hat < 1e-10].argsort()
    hessian_hat = hessian_hat[eigmax_hat < 1e-10][ind]
    psi = psi[:, eigmax_hat < 1e-10][:, ind]
    trace_hat = trace_hat[eigmax_hat < 1e-10][ind]


    n_features = np.arange(0, hessian_hat.shape[0], 50)
    n_features[0] = 1
    traces = []
    for i in n_features:
        used_hessians = hessian_hat[:i]
        optimizer = HeuristicOptimizerNegativeDefinite(used_hessians)
        w = optimizer.fit(skip_check=True)
        trace = np.trace(np.tensordot(w, used_hessians, axes=1))
        traces.append(trace)


    fig, ax = plt.subplots()
    ax.set_xlabel('number of features (sorted by trace)')
    ax.set_ylabel('trace')
    ax.plot(n_features, traces, marker='o', label='Features')
    ax.plot([1, n_features[-1]], [trace_true] * 2, color='r', linewidth=2.0,
            label='Reward function')
    ax.plot([1, n_features[-1]], [trace_hat[0]] * 2, color='g',
            linewidth=2.0, label='Best individual feature')
    ax.plot([1, n_features[-1]], [trace_hat[-1]] * 2, color='m',
            linewidth=2.0, label='Worst individual feature')
    ax.legend(loc='upper right')

    plot_state_action_function(mdp, np.dot(psi[:,:n_features[-1]], w), 'Best combination')
    plot_state_action_function(mdp, R, 'Reward')

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(copy=False)
    r_norm = scaler.fit_transform(R)
    a_norm = scaler.fit_transform(A_true)
    q_norm = scaler.fit_transform(Q_true)
    best = np.dot(psi[:, :n_features[-1]], w)
    best_norm = scaler.fit_transform(best)
    '''

    '''
    r = np.hstack([np.arange(len(eigval_true))[:, np.newaxis] + 1, eigval_true[:, np.newaxis], eigval_true_a[:, np.newaxis], eigval_hat[min_trace_idx][:, np.newaxis], eigval_hat[max_trace_idx][:, np.newaxis]])
    np.savetxt('eigen.csv', r, delimiter=',', header='x,reward function,advantage function,Feature with smallest trace,Feature with largest trace')


    print('Trace minimization...')
    print(time.strftime('%x %X %z'))
    w, hessian_trace, _ = trace_minimization(hessian_hat[:10], psi[:,:10], 0.)
    print(time.strftime('%x %X %z'))
    r_hat_trace = np.dot(psi[:, :10], w)
    r_hat_trace /= la.norm(r_hat_trace)
    trace_trace = np.trace(hessian_trace)
    eigmax_trace = la.eigh(hessian_trace)[0][-1]

    print('Max eigval minimization...')
    print(time.strftime('%x %X %z'))
    w, hessian_eigval, _ = maximum_eigenvalue_minimizarion(hessian_hat[:3], psi[:, :3], 0.)
    print(time.strftime('%x %X %z'))
    r_hat_eigval = np.dot(psi[:, :3], w)
    r_hat_eigval /= la.norm(r_hat_eigval)
    trace_eigval = np.trace(hessian_eigval)
    eigmax_eigval = la.eigh(hessian_eigval)[0][-1]

    if plot_hessians:
        fig, ax = plt.subplots()
        ax.set_xlabel('trace')
        ax.set_ylabel('max eigval')
        fig.suptitle('Hessians')
        ax.scatter(trace_hat, eigmax_hat, color='b', marker='o', label='Estimated hessians')
        ax.scatter(trace_true, eigmax_true, color='r', marker='d', s=100, label='Reward function')
        ax.scatter(trace_true_a, eigmax_true_a, color='g', marker='d', s=100, label='Advantage function')
        ax.legend(loc='upper right')
        #ax.scatter(trace_trace, eigmax_trace, color='g', marker='d', s=100)
        #ax.scatter(trace_eigval, eigmax_eigval, color='y', marker='d', s=100)


        fig, ax = plt.subplots()
        ax.set_xlabel('feature')
        ax.set_ylabel('eigval-trace')
        fig.suptitle('Hessians')
        idx = trace_hat.argsort()
        trace_hat = trace_hat[idx]
        eigmax_hat = eigmax_hat[idx]
        eigmin_hat = eigmin_hat[idx]
        ax.plot(np.arange(len(trace_hat)), trace_hat, label='Trace')
        ax.plot(np.arange(len(eigmax_hat)), eigmax_hat, label='MaxEigval')
        ax.plot(np.arange(len(eigmin_hat)), eigmin_hat, label='MinEigval')
        ax.plot(np.arange(len(trace_hat)), [trace_true]*len(trace_hat), label='True Trace')
        ax.plot(np.arange(len(eigmax_hat)),[eigmax_true]*len(trace_hat), label='True MaxEigval')
        ax.legend(loc='upper right')

    plt.show()

    '''