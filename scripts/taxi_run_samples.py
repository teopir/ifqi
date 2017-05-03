from __future__ import print_function
from ifqi.envs import TaxiEnv
from ifqi.evaluation import evaluation
from policy import BoltzmannPolicy, TaxiEnvPolicy, EpsilonGreedyBoltzmannPolicy
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

def estimate_hessian(dataset, n_episodes, policy_gradient, policy_hessian, reward_features, state_space, action_space, gamma, horizon):

    if np.ndim(reward_features) == 1:
        reward_features = reward_features[:, np.newaxis]

    n_samples = dataset.shape[0]
    n_features = reward_features.shape[1]
    n_params = policy_hessian.shape[1]
    n_states, n_actions = len(state_space), len(action_space)

    episode_reward_features = np.zeros((n_episodes, n_features))

    baseline = estimate_hessian_baseline(dataset, n_episodes, policy_gradient, policy_hessian, reward_features, state_space, action_space, gamma, horizon)

    episode_policy_gradient = np.zeros((n_episodes, n_params))
    episode_policy_hessian = np.zeros((n_episodes, n_params, n_params))

    i = 0
    episode = 0
    while i < n_samples:
        s = np.argwhere(state_space == dataset[i, 0])
        a = np.argwhere(action_space == dataset[i, 1])
        index = s * n_actions + a

        d = dataset[i, 4]

        episode_reward_features[episode, :] += d * reward_features[index, :].squeeze()
        episode_policy_gradient[episode, :] += policy_gradient[index, :].squeeze()
        episode_policy_hessian[episode, :, :] += policy_hessian[index, :, :].squeeze()

        if dataset[i, -2] == 1:

            if d == 1:
                diff = horizon - 1
            else:
                diff = horizon - np.log(d) / np.log(gamma)
            disc = (gamma * d - gamma ** (horizon + 1)) / (1 - gamma)

            episode_policy_gradient[episode, :] += diff * policy_gradient[-n_actions, :].squeeze()
            episode_policy_hessian[episode, :, :] += diff * policy_hessian[-n_actions, :, :].squeeze()
            episode_reward_features[episode, :] += disc * \
                (reward_features[-n_actions, :].squeeze())

        if dataset[i, -1] == 1:
            episode += 1

        i += 1

    print(episode_policy_gradient.shape)
    print((n_episodes, 1, n_params))
    episode_hessian = episode_policy_gradient.reshape(n_episodes, 1, n_params) * episode_policy_gradient.reshape(n_episodes, 1, n_params) + episode_policy_hessian
    episode_reward_features_baseline = episode_reward_features - baseline

    return_hessians = 1. / n_episodes * np.tensordot(episode_reward_features_baseline.T,
                                                     episode_hessian, axes=1)

    return return_hessians

def estimate_gradient(dataset, n_episodes, policy_gradient, reward_features, state_space, action_space, gamma, horizon):
    if np.ndim(reward_features) == 1:
        reward_features = reward_features[:, np.newaxis]

    n_samples = dataset.shape[0]
    n_features = reward_features.shape[1]
    n_params = policy_gradient.shape[1]
    n_states, n_actions = len(state_space), len(action_space)

    episode_reward_features = np.zeros((n_episodes, n_features))
    episode_gradient = np.zeros((n_episodes, n_params))

    baseline = estimate_gradient_baseline(dataset, n_episodes, policy_gradient, reward_features, state_space, action_space, gamma, horizon)

    i = 0
    episode = 0
    while i < n_samples:
        s = np.argwhere(state_space == dataset[i, 0])
        a = np.argwhere(action_space == dataset[i, 1])
        index = s * n_actions + a

        d = dataset[i, 4]

        episode_reward_features[episode, :] += d * reward_features[index, :].squeeze()
        episode_gradient[episode, :] +=  policy_gradient[index, :].squeeze()

        if dataset[i, -2] == 1:
            if d == 1:
                diff = horizon - 1
            else:
                diff = horizon - np.log(d) / np.log(gamma)
            disc = (gamma * d - gamma ** (horizon + 1)) / (1 - gamma)

            episode_gradient[episode, :] += diff * policy_gradient[-n_actions, :].squeeze()
            episode_reward_features[episode, :] += disc * \
                (reward_features[-n_actions, :].squeeze())

        if dataset[i, -1] == 1:
            episode += 1

        i += 1

    episode_reward_features_baseline = episode_reward_features - baseline

    return_gradient = 1. / n_episodes * np.dot(episode_reward_features_baseline.T,
                                                     episode_gradient)

    return return_gradient

def estimate_gradient_baseline(dataset, n_episodes, policy_gradient, reward_features, state_space, action_space, gamma, horizon):
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

        if dataset[i, -2] == 1:
            if d == 1:
                diff = horizon - 1
            else:
                diff = horizon - np.log(d) / np.log(gamma)
            disc = (gamma * d - gamma ** (horizon + 1)) / (1 - gamma)

            episode_gradient[episode, :] += diff * policy_gradient[-n_actions, :].squeeze()
            episode_reward_features[episode, :] += disc * \
                (reward_features[-n_actions, :].squeeze())

        if dataset[i, -1] == 1:
            vectorized_gradient = episode_gradient[episode, :].ravel()
            numerator += episode_reward_features[episode, :] * la.norm(vectorized_gradient) ** 2
            denominator += la.norm(vectorized_gradient) ** 2
            episode += 1

        i += 1


    baseline = numerator / denominator

    return baseline

def estimate_hessian_baseline(dataset, n_episodes, policy_gradient, policy_hessian, reward_features, state_space, action_space, gamma, horizon):
    n_samples = dataset.shape[0]
    n_features = reward_features.shape[1]
    n_params = policy_hessian.shape[1]
    n_states, n_actions = len(state_space), len(action_space)

    episode_reward_features = np.zeros((n_episodes, n_features))
    numerator = denominator = 0.

    episode_policy_gradient = np.zeros((n_episodes, n_params))
    episode_policy_hessian = np.zeros((n_episodes, n_params, n_params))

    i = 0
    episode = 0
    while i < n_samples:
        s = np.argwhere(state_space == dataset[i, 0])
        a = np.argwhere(action_space == dataset[i, 1])
        index = s * n_actions + a

        d = dataset[i, 4]

        episode_reward_features[episode, :] += d * reward_features[index, :].squeeze()
        episode_policy_gradient[episode, :] += policy_gradient[index, :].squeeze()
        episode_policy_hessian[episode, :, :] += policy_hessian[index, :, :].squeeze()

        if dataset[i, -2] == 1:

            if d == 1:
                diff = horizon - 1
            else:
                diff = horizon - np.log(d) / np.log(gamma)
            disc = (gamma * d - gamma ** (horizon + 1)) / (1 - gamma)

            episode_policy_gradient[episode, :] += diff * policy_gradient[-n_actions, :].squeeze()
            episode_policy_hessian[episode, :, :] += diff * policy_hessian[-n_actions, :, :].squeeze()
            episode_reward_features[episode, :] += disc * reward_features[-n_actions, :].squeeze()


        if dataset[i, -1] == 1:
            episode_hessian = np.outer(episode_policy_gradient[episode], episode_policy_gradient[episode]) + episode_policy_hessian[episode]
            vectorized_hessian = episode_hessian.ravel()
            numerator += episode_reward_features[episode, :] * la.norm(vectorized_hessian) ** 2
            denominator += la.norm(vectorized_hessian) ** 2
            episode += 1

        i += 1

    baseline = numerator / denominator

    return baseline



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
        state_features_binary = np.vstack([state_features_binary, np.zeros((1, state_features_binary.shape[1]))])
        state_features_binary = np.hstack([state_features_binary, np.eye(mdp.nS + 1, 1, -mdp.nS)])
        return state_features_binary

def fit_maximum_likelihood_boltzmann_policy(state_features, optimal_action):
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
                            tol=1e-10,
                            C=np.inf,
                            solver='newton-cg',
                            fit_intercept=False,
                            intercept_scaling=1,
                            max_iter=300,
                            multi_class='multinomial',
                            verbose=0,
                            n_jobs=1)

    lr.fit(state_features, np.concatenate([optimal_action, [0]]))
    action_weights = lr.coef_
    return action_weights

def fit_maximum_likelihood_policy_from_trajectories(state_features,
                                                      state_space,
                                                      action_space,
                                                      trajectories,
                                                      policy,
                                                      initial_parameter,
                                                      max_iter=100,
                                                      learning_rate=0.01):
    n_trajectories = int(trajectories[:, -1].sum())
    n_samples = trajectories.shape[0]
    n_states, n_actions = len(state_space), len(action_space)
    indexes = []
    for i in range(n_samples):
        s = np.argwhere(state_space == trajectories[i, 0])
        a = np.argwhere(action_space == trajectories[i, 1])
        indexes.append(s * n_actions + a)

    parameter = initial_parameter
    ite = 0
    i = 0
    while ite < max_iter:
        ite += 1
        policy.set_parameter(parameter, build_hessian=False)
        G = policy.gradient_log()

        #Gradient computation
        gradient = 0.
        while trajectories[i, -1] == 0:
            gradient += G[indexes[i]]
            i += 1
        gradient += G[indexes[i]]
        gradient /= -n_trajectories
        gradient = gradient.ravel()[:, np.newaxis]
        i = (i + 1) % n_samples

        parameter = parameter - learning_rate * gradient

    policy.set_parameter(parameter)
    return policy

def fit_maximum_likelihood_policy_from_expert_policy(state_features,
                                                      state_space,
                                                      action_space,
                                                      expert_policy,
                                                      policy,
                                                      initial_parameter,
                                                      max_iter=100,
                                                      learning_rate=0.01):
    expert_pi = policy.pi.ravel()[:, np.newaxis]
    n_states, n_actions = len(state_space), len(action_space)

    parameter = initial_parameter
    ite = 0
    while ite < max_iter:
        ite += 1

        policy.set_parameter(parameter, build_hessian=False)
        G = policy.gradient_log()

        # Gradient computation
        gradient = -1. / n_states * (expert_pi * G).sum(axis=0)
        gradient = gradient.ravel()[:, np.newaxis]
        print(gradient)
        parameter = parameter - learning_rate * gradient

    policy.set_parameter(parameter)
    return policy

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

    tol = 1e-24
    mdp = TaxiEnv()
    mdp.horizon = 100
    n_episodes = 1000

    mdp_wrap = DiscreteMdpWrapper(mdp, episodic=True)
    n_states, n_actions = mdp_wrap.nS, mdp_wrap.nA

    print('Computing expert deterministic policy...')
    expert_deterministic_policy = TaxiEnvPolicy()
    expert_deterministic_pi = expert_deterministic_policy.pi

    print('Building state features...')
    state_features = build_state_features(mdp, binary=True)
    n_parameters = state_features.shape[1]

    print('Fitting eps-boltz expert policy...')
    action_weights = fit_maximum_likelihood_boltzmann_policy(state_features,
                     expert_deterministic_policy.pi.argmax(axis=1))
    expert_policy = EpsilonGreedyBoltzmannPolicy(0.05, state_features,
                                                 action_weights)
    d_kl_det_expert = kullback_leibler_divergence(
        expert_deterministic_policy.pi, expert_policy.pi[:n_states - 1])
    print('KL-divergence expert deterministic - expert eps-boltz %s' % d_kl_det_expert)

    print('Collecting %s trajectories with expert policy...' % n_episodes)
    trajectories_ex = evaluation.collect_episodes(mdp, expert_policy, n_episodes)
    n_samples_ex = trajectories_ex.shape[0]
    print('Dataset made of %s samples' % n_samples_ex)

    action_weights = np.zeros((n_actions, n_parameters))
    ml_policy = EpsilonGreedyBoltzmannPolicy(0.05, state_features, action_weights)
    print('Fitting Maximum Likelihood eps-boltz policy from trajectories...')
    ml_policy = fit_maximum_likelihood_policy_from_trajectories(state_features,
                                                    mdp_wrap.state_space,
                                                    mdp_wrap.action_space,
                                                    trajectories_ex,
                                                    ml_policy,
                                                    action_weights.ravel()[:, np.newaxis],
                                                    max_iter=500,
                                                    learning_rate=1000.)
    d_kl_det_ml = kullback_leibler_divergence(expert_deterministic_policy.pi,
                                              ml_policy.pi[:n_states - 1])
    print('KL-divergence expert deterministic - maximum likelihood %s' % d_kl_det_ml)

    print('Collecting %s trajectories with ML policy...' % n_episodes)
    trajectories_ml = evaluation.collect_episodes(mdp, ml_policy, n_episodes)
    n_samples_ml = trajectories_ml.shape[0]
    print('Dataset made of %s samples' % n_samples_ml)

    estimator_ml = DiscreteEnvSampleEstimator(trajectories_ml,
                               mdp_wrap.gamma,
                               mdp_wrap.state_space,
                               mdp_wrap.action_space,
                               mdp_wrap.horizon)

    estimator_ex = DiscreteEnvSampleEstimator(trajectories_ex,
                               mdp_wrap.gamma,
                               mdp_wrap.state_space,
                               mdp_wrap.action_space,
                               mdp_wrap.horizon)

    print('J ML policy %s' % estimator_ml.get_J())
    print('J expert policy %s' % estimator_ex.get_J())

    # ---------------------------------------------------------------------------

    policy = expert_policy
    mdp_wrap.set_policy(policy.pi2, fix_episodic=False)
    G = policy.gradient_log()

    count_sa_hat = estimator_ex.get_count_sa()
    d_s_hat = estimator_ex.get_d_s_mu()
    d_sa_hat = estimator_ex.get_d_sa_mu()
    d_sa = mdp_wrap.compute_d_sa_mu()
    D = np.diag(d_sa)
    D_hat = np.diag(d_sa_hat)

    mdp_wrap.set_policy(policy.pi2)
    Q = mdp_wrap.compute_Q_function()
    V = mdp_wrap.compute_V_function()
    A = Q - np.repeat(V, n_actions)
    R = mdp_wrap.R

    G_norm = np.sqrt(la.multi_dot([G.T, D_hat, G]).diagonal())
    Q_norm = np.sqrt(la.multi_dot([Q.T, D_hat, Q]))
    GQ_dot = la.multi_dot([G.T, D_hat, Q])
    GQ_cos = GQ_dot / (G_norm * Q_norm + 1e-24)
    GQ_cos_norm_inf = la.norm(GQ_cos, np.inf) #Note: the expert is likely to be suboptimal

    sa_idx = count_sa_hat.nonzero()
    X = np.dot(G[sa_idx].T, np.diag(d_sa_hat[sa_idx]))

    print('Computing ECO-Qs...')
    phi = la2.nullspace(X, criterion='tol')
    print('%s ECO-Qs found' % phi.shape[1])

    print('Computing model-based ECO-Rs')
    Z = np.dot(np.eye(sa_idx[0].shape[0]) - mdp_wrap.gamma * np.dot(estimator_ex.P[sa_idx], policy.pi2[:, sa_idx].squeeze()), phi)
    psi_mb = la2.range(Z)
    print('%s model-based ECO-Rs found' % psi_mb.shape[1])

    print('Computing model-free ECO-Rs')
    pi_tilde = np.repeat(policy.pi2, n_actions, axis=0)
    Y = np.dot(np.eye(sa_idx[0].shape[0]) - pi_tilde[sa_idx][:, sa_idx].squeeze(), phi)
    psi = la2.range(Y)
    print('%s model-free ECO-Rs found' % psi.shape[1])

    psi_padded = np.zeros((n_states * n_actions, psi.shape[1]))
    psi_padded_mb = np.zeros((n_states * n_actions, psi_mb.shape[1]))
    for i in range(psi.shape[0]):
        psi_padded[sa_idx[0][i], :] = psi[i, :]
        psi_padded_mb[sa_idx[0][i], :] = psi_mb[i, :]

    # --------------------------------------------------------------------------P
    # Standard IRL algorithms on natural features
    from reward_space.inverse_reinforcement_learning.lpal import LPAL
    from reward_space.inverse_reinforcement_learning.linear_irl import LinearIRL
    from reward_space.inverse_reinforcement_learning.maximum_entropy_irl import MaximumEntropyIRL

    # Standard features
    state_action_features = np.repeat(state_features, n_actions, axis=0)
    state_action_features = np.hstack([state_action_features, np.tile(np.eye(n_actions), (n_states, 1))])

    # LPAL
    lpal = LPAL(state_action_features,
                trajectories_ex,
                estimator_ex.P,
                mdp_wrap.mu,
                mdp_wrap.gamma,
                mdp_wrap.horizon)

    lpal_policy = lpal.fit()
    from policy import TabularPolicy
    lpal_tab_policy = TabularPolicy(lpal_policy)

    print('Collecting %s trajectories with LPAL policy...' % n_episodes)
    trajectories_lpal = evaluation.collect_episodes(mdp, lpal_tab_policy, n_episodes)
    n_samples_lpal = trajectories_lpal.shape[0]
    print('Dataset made of %s samples' % n_samples_lpal)

    estimator_lpal = DiscreteEnvSampleEstimator(trajectories_lpal,
                               mdp_wrap.gamma,
                               mdp_wrap.state_space,
                               mdp_wrap.action_space,
                               mdp_wrap.horizon)

    print('J LPAL policy %s' % estimator_lpal.get_J())

    #Linear IRL
    transition_model = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            for s1 in range(n_states):
                transition_model[s, a, s1] = estimator_ex.P[s * n_actions + a, s1]
    linear_irl = LinearIRL(transition_model,
                           expert_policy.pi,
                           mdp_wrap.gamma,
                           r_max=1.,
                           l1_penalty=0.,
                           type_='state')
    linear_irl_reward = linear_irl.fit()
    linear_irl_reward = np.repeat(linear_irl_reward, n_actions)


    #Maximum entropy
    me_state = MaximumEntropyIRL(state_features,
                                 trajectories_ex,
                                 transition_model,
                                 mdp_wrap.mu,
                                 mdp_wrap.gamma,
                                 mdp_wrap.horizon,
                                 learning_rate=0.01,
                                 max_iter=100)
    me_reward = me_state.fit()
    me_reward = np.repeat(me_reward, n_actions)

    # ---------------------------------------------------------------------------

    names = []
    basis_functions = []
    gradients = []
    hessians = []
    H = policy.hessian_log()

    # Hessian estimation - model free

    print('Estimating hessians model free...')

    hessian_hat = estimate_hessian(trajectories_ex, n_episodes, G, H, psi_padded, \
                                   mdp_wrap.state_space, mdp_wrap.action_space, mdp_wrap.gamma, mdp_wrap.horizon)

    eigval_hat, _ = la.eigh(hessian_hat)
    eigmax_hat, eigmin_hat = eigval_hat[:, -1], eigval_hat[:, 0]
    trace_hat = np.trace(hessian_hat, axis1=1, axis2=2)

    #Heuristic for negative semidefinite
    neg_idx = np.argwhere(eigmax_hat < -eigmin_hat / 10).ravel()
    hessian_hat_neg = hessian_hat[neg_idx]

    '''
    #HEURISTIC SOLUTION
    #max eig minimization or hessian heuristic in this case are
    #the same since the dimension of parameter space is 1
    '''

    optimizer = HeuristicOptimizerNegativeDefinite(hessian_hat_neg)
    w = optimizer.fit(skip_check=True)
    eco_r = np.dot(psi_padded[:, neg_idx], w)
    #Penalization
    eco_r[np.setdiff1d(np.arange(eco_r.shape[0]), sa_idx[0])] = min(eco_r)

    # ---------------------------------------------------------------------------
    # Hessian estimation - model based

    print('Estimating hessians model based...')

    hessian_hat = estimate_hessian(trajectories_ex, n_episodes, G, H, psi_padded_mb, \
                                   mdp_wrap.state_space, mdp_wrap.action_space, mdp_wrap.gamma, mdp_wrap.horizon)

    eigval_hat, _ = la.eigh(hessian_hat)
    eigmax_hat, eigmin_hat = eigval_hat[:, -1], eigval_hat[:, 0]
    trace_hat = np.trace(hessian_hat, axis1=1, axis2=2)

    #Heuristic for negative semidefinite
    neg_idx = np.argwhere(eigmax_hat < -eigmin_hat / 10).ravel()
    hessian_hat_neg = hessian_hat[neg_idx]

    '''
    #HEURISTIC SOLUTION
    #max eig minimization or hessian heuristic in this case are
    #the same since the dimension of parameter space is 1
    '''

    optimizer = HeuristicOptimizerNegativeDefinite(hessian_hat_neg)
    w = optimizer.fit(skip_check=True)
    eco_r_mb = np.dot(psi_padded_mb[:, neg_idx], w)
    #Penalization
    eco_r_mb[np.setdiff1d(np.arange(eco_r.shape[0]), sa_idx[0])] = min(eco_r)

    # ---------------------------------------------------------------------------
    # Build feature sets
    names.append('Reward function')
    basis_functions.append(R)
    names.append('Advantage function')
    basis_functions.append(A)
    names.append('ECO-R heuristic model free')
    basis_functions.append(eco_r)
    names.append('ECO-R heuristic model based')
    basis_functions.append(eco_r_mb)
    names.append('Linear irl - Russell Ng')
    basis_functions.append(linear_irl_reward)
    names.append('Maximum entropy natural features')
    basis_functions.append(me_reward)

    '''
    #Gradient and hessian estimation
    '''

    # Rescale rewards into to have difference between max and min equal to 1
    scaler = RangeScaler()

    scaled_basis_functions = []
    for bf in basis_functions:
        sbf = scaler.fit_transform(bf[:, np.newaxis]).ravel()
        scaled_basis_functions.append(sbf)

    gradients, hessians, gradient_norms2, gradient_normsinf, eigvals, traces, eigvals_max = [], [], [], [], [], [], []
    print('Estimating gradient and hessians...')
    for sbf in scaled_basis_functions:
        gradient = estimate_gradient(trajectories_ex, n_episodes,G, sbf, \
                                mdp_wrap.state_space, mdp_wrap.action_space, mdp_wrap.gamma, mdp_wrap.horizon)[0]
        hessian = estimate_hessian(trajectories_ex, n_episodes, G, H, sbf, \
                              mdp_wrap.state_space, mdp_wrap.action_space, mdp_wrap.gamma, mdp_wrap.horizon)[0]
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

    if plot:
        fig, ax = plt.subplots()
        ax.set_xlabel('trace')
        ax.set_ylabel('max eigval')
        fig.suptitle('Hessians')
        ax.scatter(trace_hat, eigmax_hat, color='k', marker='+',
                   label='Estimated hessians')
        for i in range(len(names)):
            ax.plot(traces_np[i], eigvals_max_np[i], marker='o', label=names[i])
        ax.legend(loc='upper right')

        _range = np.arange(len(eigvals_np[0]))
        fig, ax = plt.subplots()
        ax.set_xlabel('index')
        ax.set_ylabel('eigenvalue')
        fig.suptitle('Hessian Eigenvalues')
        for i in range(len(names)):
            ax.plot(_range, eigvals_np[i], marker='+', label=names[i])

        ax.legend(loc='upper right')
        plt.yscale('symlog', linthreshy=1e-6)

    '''
    #REINFORCE training
    '''

    def reward_function(sbf, traj):
        diff = mdp_wrap.horizon - traj.shape[0]
        disc = (1 - mdp_wrap.gamma ** (diff + 1)) / (1 - mdp_wrap.gamma)
        abs_reward = disc * sbf[-n_actions, ]
        rewards = sbf[(traj[:, 0] * n_actions + traj[:, 1]).astype(int)]
        rewards[-1] += abs_reward
        return rewards


    learner = PolicyGradientLearner(mdp, policy, lrate=0.1, verbose=1,
                                    max_iter_opt=200, tol_opt=-1., tol_eval=0.,
                                    estimator='reinforce')

    theta0 = np.zeros((240, 1))

    histories = []
    for i in range(len(scaled_basis_functions)):
        print(names[i])
        sbf = scaled_basis_functions[i]
        #theta, history = learner.optimize(theta0, reward=lambda traj: sbf[
        #    (traj[:, 0] * 6 + traj[:, 1]).astype(int)],return_history=True)
        theta, history = learner.optimize(theta0, reward=lambda traj: reward_function(sbf, traj),return_history=True)

        histories.append(history)

    histories = np.array(histories)

    t = PrettyTable()
    t.add_column('Basis function', names)
    t.add_column('Final return', histories[:, -1, 1])
    print(t)

    plot=True
    if plot:
        _range = np.arange(201)
        fig, ax = plt.subplots()
        ax.set_xlabel('average return')
        ax.set_ylabel('iterations')
        fig.suptitle('REINFORCE - Average return')

        ax.plot([0, 200], [estimator_ex.get_J(), estimator_ex.get_J()], color='k',
                label='Optimal return')
        for i in range(len(names)):
            ax.plot(_range, histories[i, :, 1], marker='+', label=names[i])

        ax.legend(loc='upper right')

    saveme = np.zeros(2, dtype=object)
    saveme[0] = names
    saveme[1] = histories
    np.save('data/taxi_comparision_%s' % mytime, saveme)
