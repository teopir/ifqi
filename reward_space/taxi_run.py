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
#import cvxpy
import time

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

    n_samples = dataset.shape[0]
    n_features = reward_features.shape[1]
    n_params = policy_hessian.shape[1]
    n_states, n_actions = len(state_space), len(action_space)

    episode_reward_features = np.zeros((n_episodes, n_features))
    episode_hessian = np.zeros((n_episodes, n_params, n_params))

    #baseline = estimate_hessian_baseline(dataset, n_episodes, policy_gradient, policy_hessian, reward_features, state_space, action_space)

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

    episode_reward_features_baseline = episode_reward_features #- baseline

    return_hessians = 1. / n_episodes * np.tensordot(episode_reward_features_baseline.T,
                                                     episode_hessian, axes=1)

    return return_hessians

def estimate_gradient(dataset, n_episodes, policy_gradient, reward_features, state_space, action_space):

    n_samples = dataset.shape[0]
    n_features = reward_features.shape[1]
    n_params = policy_gradient.shape[1]
    n_states, n_actions = len(state_space), len(action_space)

    episode_reward_features = np.zeros((n_episodes, n_features))
    episode_gradient = np.zeros((n_episodes, n_params))

    #baseline = estimate_gradient_baseline(dataset, n_episodes, policy_gradient, reward_features, state_space, action_space)

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

    episode_reward_features_baseline = episode_reward_features #- baseline

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
                            tol=1e-5,
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

if __name__ == '__main__':

    plot = False
    plot_gradient = True
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

    tol = 1e-24
    mdp = TaxiEnv()
    mdp.horizon = 100
    n_episodes = 1000

    print('Computing optimal policy...')
    opt_policy = TaxiEnvPolicy()
    pi_opt = opt_policy.PI


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


    from policy_gradient.policy_gradient_learner import PolicyGradientLearner
    learner = PolicyGradientLearner(mdp, policy, lrate=1., verbose=1, max_iter_opt=25, tol_opt=0., tol_eval=0.)
    #theta0 = policy.state_action_parameters
    #theta0 = policy.state_action_parameters + 10. * np.random.randn(n_parameters, 1)
    #theta0 = np.zeros((n_parameters, 1))
    theta0 = np.concatenate([np.zeros(5 * 39),  np.ones(1 * 39)])[:, np.newaxis]
    theta = learner.optimize(theta0)
    policy.set_parameter(theta)


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
    D = np.diag(d_sa_mu )
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
        D_idx = D.copy()
        D_idx[D_idx > 0] = 1
        s = la.svd(np.dot(G.T, D), compute_uv=False)
        plt.plot(np.arange(len(s)), s, marker='o', label='Gradient * d(s,a)')
    '''
    #---------------------------------------------------------------------------
    #Q-function and A-function estimation with PVFs
    print('-' * 100)

    print('Q-function and A-function estimation with pvf')
    ks = np.arange(kmin, kmax, kstep)
    ks[0] = 1

    errors_pvf = []
    Q_hat_pvf = []
    pvfs = []
    eigvals_pvf = []
    errors_prf = []

    scorer = Scorer(d_sa_mu / sum(d_sa_mu))

    if perform_estimation_pvf and plot_pvf:
        fig, ax = plt.subplots()
        ax.set_xlabel('number of features')
        ax.set_ylabel(scorer.__str__())
        fig.suptitle('Q-function approx with PVF')

        fig2, ax2 = plt.subplots()
        ax2.set_xlabel('number of features')
        ax2.set_ylabel(scorer.__str__())
        fig2.suptitle('A-function approx with PVF')

    for method in methods:
        print('Fitting ProtoValue Functions %s ...' % method)
        pvf_estimator = ProtoValueFunctionsEstimator(mdp_wrap.state_space,
                                                     mdp_wrap.action_space,
                                                     'norm-laplacian',
                                                     method)
        pvf_estimator.fit(dataset)
        eigval, pvf = pvf_estimator.transform(kmax)
        eigvals_pvf.append(eigval)
        pvfs.append(pvf)

        print('Q-function estimation PVF')
        Q_hat, _, _, _ = la2.lsq(pvf, Q_true)
        print(scorer.score(Q_true, Q_hat))

        #Compure and rank prf
        prf = np.dot(np.eye(mdp_wrap.nA * mdp_wrap.nS) - pi_tilde, pvf)
        eigval_prf = eigval / la.norm(prf, axis=0)
        rank = eigval_prf.argsort()
        prf = prf[:, rank]
        prf = prf / la.norm(prf, axis=0)

        print('A-function estimation PRF')
        A_hat, _, _, _ = la2.lsq(prf, A_true)
        print(scorer.score(A_true, A_hat))

        if perform_estimation_pvf:
            q_hat, error_q = estimate(pvf, Q_true, scorer, ks)
            a_hat, error_a = estimate(prf, A_true, scorer, ks)
            Q_hat_pvf.append(q_hat)
            errors_pvf.append(error_q)
            errors_prf.append(error_a)

            if plot_pvf:
                ax.plot(ks, errors_pvf[-1], marker='o', label='PVF ' + method)
                ax.legend(loc='upper right')

                ax2.plot(ks, errors_prf[-1], marker='o', label='PRF ' + method)
                ax2.legend(loc='upper right')

    #---------------------------------------------------------------------------
    #Q-function and A-function estimation with GPVFs
    print('-' * 100)

    print('Q-function and A-function estimation with gpvf')

    errors_gpvf = []
    Q_hat_gpvf = []
    gpvfs = []
    errors_gprf = []
    eigvals_gpvf = []

    if perform_estimation_gpvf and plot_gpvf and not plot_pvf:
        fig, ax = plt.subplots()
        ax.set_xlabel('number of features')
        ax.set_ylabel(scorer.__str__())
        fig.suptitle('Q-function approx with GPVF')

        fig2, ax2 = plt.subplots()
        ax2.set_xlabel('number of features')
        ax2.set_ylabel(scorer.__str__())
        fig2.suptitle('A-function approx with GPVF')

    for i in range(len(methods)):
        print('Fitting ProtoReward Functions %s ...' % methods[i])
        G_basis = la2.range(G)
        projection_matrix = np.eye(mdp_wrap.nA * mdp_wrap.nS) - la.multi_dot([G_basis, la.inv(la.multi_dot([G_basis.T, D, G_basis])), G_basis.T, D])
        gpvf = np.dot(projection_matrix, pvfs[i])

        #Re rank the gpvfs
        eigval = eigvals_pvf[i] / la.norm(gpvf, axis=0)
        rank = eigval.argsort()
        eigval = eigval[rank]
        gpvf = gpvf[:, rank]
        gpvf = gpvf / la.norm(gpvf, axis=0)
        gpvfs.append(gpvf)
        eigvals_gpvf.append(eigval)

        print('Q-function estimation GPVF')
        Q_hat, _, _, _ = la2.lsq(gpvf, Q_true)
        print(scorer.score(Q_true, Q_hat))

        # Compute and rank prf
        gprf = np.dot(np.eye(mdp_wrap.nA * mdp_wrap.nS) - pi_tilde, gpvf)
        eigval_gprf = eigval / la.norm(gprf, axis=0)
        rank = eigval_gprf.argsort()
        gprf = gprf[:, rank]
        gprf = gprf / la.norm(gprf, axis=0)

        print('A-function estimation PRF')
        #A_hat, _, _, _ = la2.lsq(gprf, A_true)
        #print(scorer.score(A_true, A_hat))


        if perform_estimation_gpvf:
            q_hat, error_q = estimate(gpvf, Q_true, scorer, ks)
            a_hat, error_a = estimate(gprf, A_true, scorer, ks)
            Q_hat_gpvf.append(q_hat)
            errors_gpvf.append(error_q)
            errors_gprf.append(error_a)

            if plot_gpvf:
                ax.plot(ks, errors_gpvf[-1], marker='o', label='GPVF ' + method)
                ax.legend(loc='upper right')

                ax2.plot(ks, errors_gprf[-1], marker='o', label='GPRF ' + method)
                ax2.legend(loc='upper right')


    '''
    '''
    print('-' * 100)

    print('Computing Q-function approx space...')
    X = np.dot(G.T, D_hat)
    phi = la2.nullspace(X)

    psi = phi * la.norm(R)
    #print('Computing reward function approx space...')
    #Y = np.dot(np.eye(mdp_wrap.nA * mdp_wrap.nS) - pi_tilde, phi)
    #psi = la2.range(Y) *10000

    #---------------------------------------------------------------------------
    #Hessian estimation

    print('Estimating hessians...')
    H = policy.hessian_log()

    a_true = A_true / la.norm(A_true) * la.norm(R)
    r_true = R

    hessian_true = estimate_hessian(dataset, n_episodes, G, H, r_true[:, np.newaxis], \
                                    mdp_wrap.state_space, mdp_wrap.action_space)[0]

    hessian_true_a = estimate_hessian(dataset, n_episodes, G, H, a_true[:, np.newaxis], \
                                    mdp_wrap.state_space, mdp_wrap.action_space)[0]

    hessian_hat = estimate_hessian(dataset, n_episodes, G, H, psi, \
                                    mdp_wrap.state_space, mdp_wrap.action_space)




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

    from inverse_reinforcement_learning.hessian_optimization import HeuristicOptimizerNegativeDefinite
    ho = HeuristicOptimizerNegativeDefinite(hessian_hat[eigmax_hat < 1e-10])
    w = ho.fit()
    best_hessian = np.tensordot(w, hessian_hat[eigmax_hat < 1e-10], axes=1)

    trace_best = np.trace(best_hessian)
    eigval_best, _ = la.eigh(best_hessian)
    eigmax_best = eigval_best[-1]

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

        ax.legend(loc='upper right')
        plt.yscale('symlog', linthreshy=1e-8)

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