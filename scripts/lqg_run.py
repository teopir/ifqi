from __future__ import print_function
from ifqi.envs import LQG1D
from ifqi.evaluation import evaluation
from policy import GaussianPolicy1D
from reward_space.utils.continuous_env_sample_estimator import ContinuousEnvSampleEstimator
from reward_space.utils.k_neighbors_regressor_2 import KNeighborsRegressor2
from reward_space.inverse_reinforcement_learning.hessian_optimization import HeuristicOptimizerNegativeDefinite
#from reward_space.inverse_reinforcement_learning.maximum_entropy_irl import MaximumEntropyIRL
from reward_space.policy_gradient.policy_gradient_learner import PolicyGradientLearner
from reward_space.policy_gradient.gradient_estimator import MaximumLikelihoodEstimator
from reward_space.proto_value_functions.proto_value_functions_estimator import ContinuousProtoValueFunctions
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from prettytable import PrettyTable
import copy
import time
import numpy as np
import numpy.linalg as la
import reward_space.utils.linalg2 as la2
import matplotlib.pyplot as plt

def plot_state_action_function(f, title, states=None, actions=None, _cmap='coolwarm'):
    if states is None:
        states = np.arange(-10, 11, .5)
    if actions is None:
        actions = np.arange(-8, 9, .5)
    states, actions = np.meshgrid(states, actions)
    z = f(states.ravel(), actions.ravel()).reshape(states.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(states, actions, z, rstride=1, cstride=1,
                    cmap=plt.get_cmap(_cmap),
                    linewidth=0.3, antialiased=True)
    ax.set_title(title)

def get_knn_function_for_plot(knn, rescale=False):
    if rescale:
        return lambda states, actions: knn.predict \
         (np.hstack([states[:, np.newaxis], actions[:, np.newaxis]]), rescale=False)
    else:
        return lambda states, actions: knn.predict \
            (np.hstack([states[:, np.newaxis], actions[:, np.newaxis]]))

def get_knn_function_for_prediction(knn, rescale=False):
    if rescale:
        return lambda traj: knn.predict(traj[:, :2], rescale=False)
    else:
        return lambda traj: knn.predict(traj[:, :2])

def train(learner, states_actions, gaussian_kernel, k, penalty, feature, knn_penalty=None, plot=False):
    knn = KNeighborsRegressor(n_neighbors=k, weights=gaussian_kernel)
    knn.fit(states_actions, feature.ravel())

    if plot:
        if penalty:
            function = lambda states, actions: .5 * \
                        get_knn_function_for_plot(knn)(states, actions) *+ \
                        .5 * 1. / 2. * get_knn_function_for_plot(knn_penalty, True)(states, actions)
        else:
            function = get_knn_function_for_plot(knn)
        plot_state_action_function(function, '%sknn %s' % (k, '- penalty' if penalty else ''))

    if penalty:
        function = lambda traj: .5 * get_knn_function_for_prediction(knn)(traj) + \
                             .5 * 1. / 2. * get_knn_function_for_prediction(knn_penalty, True)(traj)
    else:
        function = get_knn_function_for_prediction(knn)

    theta, history = learner.optimize(-0.2, reward=function, return_history=True)
    return history


def plot_curve(mdp, policy, reward=None, edgecolor='#3F7F4C', facecolor='#7EFF99'):
    policy = copy.deepcopy(policy)
    k_list = np.arange(-1.5, -.1, 0.01)
    episodes = 50
    means = []
    stds = []
    for k in k_list:
        policy.set_parameter(k)
        returns = []
        for i in range(episodes):
            dataset = evaluation.collect_episodes(mdp, policy, 1)
            if reward is None:
                returns.append(np.dot(dataset[:, 2], dataset[:, 4]))
            else:
                returns.append(np.dot(reward(dataset), dataset[:, 4]))
        means.append(np.mean(returns))
        stds.append(np.std(returns))

    means = np.array(means)
    stds = np.array(stds)
    means -= max(means)


    ax.set_xlabel('k')
    ax.set_ylabel('J(k)')

    ax.plot(k_list, means, 'k', color=edgecolor, marker='+')
    ax.fill_between(k_list, means - stds, means + stds,
                    alpha=1, edgecolor=edgecolor, facecolor=facecolor,
                    linewidth=0)


if __name__ == '__main__':

    mytime = time.time()
    plot = False
    plot_train = False

    n_episodes = 20

    #number of neighbors for kernel extension
    k_neighbors = [1, 2, 5, 10, 50]
    #k_neighbors = [1]
    penalty_list = [True, False]

    k_pvf = []

    mdp = LQG1D()

    #Policy parameters
    K = mdp.computeOptimalK()
    print(K)
    sigma = np.sqrt(1.)

    policy = GaussianPolicy1D(K, sigma)

    reward_function = lambda states, actions: np.array(map(lambda s, a: \
                                -mdp.get_cost(s,a), states, actions))
    Q_function = lambda states, actions: np.array(map(lambda s, a: \
                 mdp.computeQFunction(s, a, K, sigma ** 2), states, actions))
    V_function = lambda states: np.array(map(lambda s: \
                 mdp.computeVFunction(s, K, sigma ** 2), states))
    A_function = lambda states, actions: Q_function(states, actions) - V_function(states)

    if plot:
        plot_state_action_function(reward_function, 'Reward')
        plot_state_action_function(Q_function, 'Q-function')
        plot_state_action_function(A_function, 'A-function')
        plot_state_action_function(lambda s, a: V_function(s), 'V-function')

    #Collect samples
    dataset = evaluation.collect_episodes(mdp, policy, n_episodes)
    n_samples = dataset.shape[0]


    estimator = ContinuousEnvSampleEstimator(dataset, mdp.gamma)
    ml_estimator = MaximumLikelihoodEstimator(dataset)
    d_sa_mu_hat = estimator.get_d_sa_mu()
    D_hat = np.diag(d_sa_mu_hat)

    states_actions = dataset[:, :2]
    states = dataset[:, 0]
    actions = dataset[:, 1]
    rewards = dataset[:, 2]
    q_function = np.array(map(lambda s, a: mdp.computeQFunction(s, a, K, sigma ** 2), states, actions))
    v_function = np.array(map(lambda s: mdp.computeVFunction(s, K, sigma ** 2), states))
    a_function = q_function - v_function

    G = np.array(map(lambda s,a: policy.gradient_log(s,a), states, actions))
    H = np.array(map(lambda s,a: policy.hessian_log(s,a), states, actions))

    print('-' * 100)
    print('Computing Q-function approx space...')

    X = np.dot(G.T, D_hat)
    phi = la2.nullspace(X)

    print('-' * 100)
    print('Computing A-function approx space...')

    sigma_kernel = 1.
    def gaussian_kernel(x):
        return 1. / np.sqrt(2 * np.pi * sigma_kernel ** 2) * np.exp(- 1. / 2 * x ** 2 / (sigma_kernel ** 2))

    knn_states = NearestNeighbors(n_neighbors=10)
    knn_states.fit(states[:, np.newaxis])
    pi_tilde = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        pi_tilde[i, i] = policy.pdf(states[i], actions[i])
        _, idx = knn_states.kneighbors(states[i])
        idx = idx.ravel()
        for j in idx:
            pi_tilde[i, j] = policy.pdf(states[j], actions[j])

    pi_tilde /= pi_tilde.sum(axis=1)

    Y = np.dot(np.eye(n_samples) - pi_tilde, phi)
    psi = la2.range(Y)

    names = []
    basis_functions = []
    gradients = []
    hessians = []

    names.append('Reward function')
    basis_functions.append(rewards / la.norm(rewards))
    names.append('Advantage function')
    basis_functions.append(a_function / la.norm(a_function))

    print('Computing gradients and hessians...')
    gradient_hat = ml_estimator.estimate_gradient(psi, G, use_baseline=True)
    hessian_hat = ml_estimator.estimate_hessian(psi, G, H, use_baseline=True)

    '''
    Since the dimension of the parameter space is 1 if the trace and max eig are
    the same, if the hessian is positive just change the sign of the feature.
    We can always get all negative definite hessians. Thus all the features
    can be used.
    '''

    print('Minimum hessian %s' % hessian_hat.max())
    print('Maximum hessian %s' % hessian_hat.min())

    mask = (hessian_hat > 0).ravel()
    psi[:, mask] *= -1
    hessian_hat[mask] *= -1

    '''
    GIRL
    '''

    from reward_space.inverse_reinforcement_learning.girl import LinearGIRL
    lgirl = LinearGIRL(dataset, dataset[:, :2] ** 2, G)
    par_sa_squared = lgirl.fit()

    lgirl = LinearGIRL(dataset, abs(dataset[:, :2]), G)
    par_sa_linear = lgirl.fit()

    print(par_sa_squared)
    print(par_sa_linear)


    def girl1(s, a):
        return s**2 * par_sa_squared[0] + a**2 * par_sa_squared[1]

    def girl2(s, a):
        return abs(s) * par_sa_linear[0] + abs(a) * par_sa_linear[1]

    g1 = girl1(dataset[:, 0], dataset[:, 1]).ravel()
    g2 = girl2(dataset[:, 0], dataset[:, 1]).ravel()

    names.append('GIRL - square')
    basis_functions.append(g1)

    names.append('GIRL - linear')
    basis_functions.append(g2)

    '''
    TRACE MINIMIZATION
    max eig minimization or hessian heuristic in this case are
    the same since the dimension of parameter space is 1
    '''

    optimizer = HeuristicOptimizerNegativeDefinite(hessian_hat)
    w = optimizer.fit()
    grbf = np.dot(psi, w)
    names.append('ECO-R')
    basis_functions.append(grbf)



    '''
    Gradient and hessian estimation
    '''

    #Rescale rewards into to have difference between max and min equal to 1
    scaler = MinMaxScaler()

    scaled_grbf = scaler.fit_transform(grbf[:, np.newaxis]).ravel()

    scaled_basis_functions = []
    for bf in basis_functions:
        sbf = scaler.fit_transform(bf[:, np.newaxis]).ravel()
        scaled_basis_functions.append(sbf)

    print('Estimating gradient and hessians...')
    for sbf in scaled_basis_functions:
        gradients.append(ml_estimator.estimate_gradient(sbf, G, use_baseline=True))
        hessians.append(ml_estimator.estimate_hessian(sbf, G, H, use_baseline=True))

    t = PrettyTable()
    t.add_column('Basis function', names)
    t.add_column('Gradient', gradients)
    t.add_column('Hessian', hessians)
    print(t)

    print('Saving results...')
    gradients_np = np.vstack(gradients)
    hessians_np = np.vstack(hessians)

    saveme = np.zeros(3, dtype=object)
    saveme[0] = names
    saveme[1] = gradients_np
    saveme[2] = hessians_np
    np.save('data/lqg/lqg_gradients_hessians_%s_%s' % (sigma**2, mytime), saveme)

    if plot:
        fig, ax = plt.subplots()
        ax.set_xlabel('trace')
        ax.set_ylabel('maximum eigenvalue')
        fig.suptitle('Basis functions')

        ax.scatter(hessian_hat.ravel(), hessian_hat.ravel(), marker='+', color='k', label='All grbf')
        for i in range(len(basis_functions)):
            ax.scatter(hessians_np[i].ravel(), hessians_np[i].ravel(), marker='o', label=names[i])

        ax.legend(loc='upper left')

    '''
    REINFORCE training
    '''
    print('-' * 100)
    print('Estimating d(s,a)...')

    count_sa_hat = estimator.get_count_sa()
    count_sa_hat /= count_sa_hat.max()

    count_sa_knn = KNeighborsRegressor2(n_neighbors=5, weights=gaussian_kernel)
    count_sa_knn.fit(states_actions, count_sa_hat)
    plot_state_action_function(get_knn_function_for_plot(count_sa_knn, True), 'd(s,a)')

    '''
    print('-' * 100)
    print('Training with REINFORCE using the estimated grbf trace minimizer')

    learner = PolicyGradientLearner(mdp, policy, lrate=0.01, verbose=1, tol_opt=-1.,
                            lrate_decay={'method' :'inverse', 'decay': .3})

    knn_histories = []
    knn_labels = []
    for k in k_neighbors:
        for penalty in penalty_list:
            label = 'GRBF trace minimizer - %sknn%s' % (k, ' - penalty' if penalty else '')
            print(label)
            knn_labels.append(label)
            history = train(learner, states_actions, gaussian_kernel, k, penalty,
                            scaled_grbf, count_sa_knn, False)
            knn_histories.append(history)


    knn_histories = np.array(knn_histories)
    t = PrettyTable()
    t.add_column('Basis function', knn_labels)
    t.add_column('Final parameter', knn_histories[:, -1, 0])
    t.add_column('Final return', knn_histories[:, -1, 1])
    t.add_column('Final gradient', knn_histories[:, -1, 2])
    print(t)


    if plot:
        _range = np.arange(101)
        fig, ax = plt.subplots()
        ax.set_xlabel('parameter')
        ax.set_ylabel('iterations')
        fig.suptitle('REINFORCE - Parameter')

        ax.plot([0, 100], [K.ravel(), K.ravel()], color='k', label='Optimal parameter')
        for i in range(len(knn_histories)):
            ax.plot(_range, np.concatenate(knn_histories[i, :, 0]).squeeze(), marker='+', label=knn_labels[i])

        ax.legend(loc='upper right')

    saveme = np.zeros(2, dtype=object)
    saveme[0] = knn_labels
    saveme[1] = knn_histories
    np.save('data/lqg/lqg_gbrf_knn_%s_%s' % (sigma**2, mytime), saveme)

    '''
    print('-' * 100)
    print('Training with REINFORCE using true reward and true a function')

    iterations = 200

    learner = PolicyGradientLearner(mdp, policy, max_iter_opt=iterations, lrate=0.002,
            gradient_updater='adam', verbose=1, tol_opt=-1.)

    #Building knn for reward and advantage
    _states = np.arange(-10, 10.5, .5)
    _actions = np.arange(-8, 8.5, .5)
    _states, _actions = np.meshgrid(_states, _actions)
    _states_actions = np.hstack(
        [_states.ravel()[:, np.newaxis], _actions.ravel()[:, np.newaxis]])

    z = reward_function(_states.ravel(), _actions.ravel()).ravel()[:,np.newaxis]
    z = scaler.fit_transform(z)
    history_reward = train(learner, _states_actions, gaussian_kernel, 1, False, z, None, False)

    z = A_function(_states.ravel(), _actions.ravel()).ravel()[:, np.newaxis]
    z = scaler.fit_transform(z)
    history_advantage = train(learner, _states_actions, gaussian_kernel, 1, False, z, None, False)

    z = girl1(_states.ravel(), _actions.ravel()).ravel()[:, np.newaxis]
    z = scaler.fit_transform(z)
    history_g1 = train(learner, _states_actions, gaussian_kernel, 1,
                              False, z, None, False)

    z = girl2(_states.ravel(), _actions.ravel()).ravel()[:, np.newaxis]
    z = scaler.fit_transform(z)
    history_g2 = train(learner, _states_actions, gaussian_kernel, 1,
                              False, z, None, False)

    labels = ['Reward function', 'Advantage function', 'GIRL - linear', 'GIRL - square']
    histories = [history_reward, history_advantage, history_g1, history_g2]

    for i in range(4, len(scaled_basis_functions)):
        print(names[i])
        sbf = scaled_basis_functions[i]
        history = train(learner, states_actions, gaussian_kernel, 2, True, sbf, count_sa_knn, True)
        histories.append(history)
    labels = labels + map(lambda x: x + ' 2knn - penalized', names[4:])

    histories = np.array(histories)
    t = PrettyTable()
    t.add_column('Basis function', labels)
    t.add_column('Final parameter', histories[:, -1, 0])
    t.add_column('Final return', histories[:, -1, 1])
    t.add_column('Final gradient', histories[:, -1, 2])
    print(t)

    plot = True
    if plot:
        _range = np.arange(iterations+1)
        fig, ax = plt.subplots()
        ax.set_xlabel('parameter')
        ax.set_ylabel('iterations')
        fig.suptitle('REINFORCE - Parameter ' + str(sigma**2) + str(mytime))

        ax.plot([0, iterations+1], [K.ravel(), K.ravel()], color='k', label='Optimal parameter')
        for i in range(len(histories)):
            ax.plot(_range, np.concatenate(histories[i, :, 0]).squeeze(), marker=None, label=labels[i])

        ax.legend(loc='upper right')

        _range = np.arange(iterations+1)
        fig, ax = plt.subplots()
        ax.set_xlabel('parameter')
        ax.set_ylabel('iterations')
        fig.suptitle('REINFORCE - Return ' + str(sigma**2) + str(mytime))

        ax.plot([0, iterations+1], [K.ravel(), K.ravel()], color='k', label='Optimal parameter')
        for i in range(len(histories)):
            ax.plot(_range, histories[i, :, 1], marker=None, label=labels[i])

        ax.legend(loc='upper right')


    saveme = np.zeros(2, dtype=object)
    saveme[0] = labels
    saveme[1] = histories
    np.save('data/lqg/lqg_comparision_%s_%s' % (sigma**2, mytime), saveme)
