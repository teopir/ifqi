from __future__ import print_function
from ifqi.envs import LQG1D
from ifqi.evaluation import evaluation
import numpy as np
from policy import GaussianPolicy1D
from utils.continuous_env_sample_estimator import ContinuousEnvSampleEstimator
import utils.linalg2 as la2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la
from policy_gradient.gradient_estimator import MaximumLikelihoodEstimator
from sklearn.neighbors import KNeighborsRegressor
from inverse_reinforcement_learning.hessian_optimization import HeuristicOptimizerNegativeDefinite
import bottleneck as bn
from policy_gradient.policy_gradient_learner import PolicyGradientLearner
from sklearn.preprocessing import MinMaxScaler
from utils.k_neighbors_regressor_2 import KNeighborsRegressor2
from proto_value_functions.proto_value_functions_estimator import ContinuousProtoValueFunctions


def plot_state_action_function(f, title, states=None, actions=None, _cmap='coolwarm'):
    if states is None:
        states = np.arange(-10, 10, 1)
    if actions is None:
        actions = np.arange(-8, 8, 1)
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
                        get_knn_function_for_plot(knn)(states, actions) + \
                        .5 * 1./0.4 * get_knn_function_for_plot(knn_penalty, True)(states, actions)
        else:
            function = get_knn_function_for_plot(knn)
        plot_state_action_function(function, '%sknn %s' % (k, '- penalty' if penalty else ''))

    if penalty:
        function = lambda traj: .5 * get_knn_function_for_prediction(knn)(traj) + \
                            .5 * 1. / 0.4 * get_knn_function_for_prediction(knn_penalty, True)(traj)
    else:
        function = get_knn_function_for_prediction(knn)

    theta, history = learner.optimize(-0.2, reward=function, return_history=True)
    return history

if __name__ == '__main__':

    plot = False
    plot_train = True
    fast_heuristic = True

    mdp = LQG1D()

    #Policy parameters
    K = mdp.computeOptimalK()
    sigma = np.sqrt(0.1)

    policy = GaussianPolicy1D(K, sigma)

    if plot:
        reward_function = lambda states, actions: np.array(map(lambda s, a:\
                                    -mdp.get_cost(s,a), states, actions))
        Q_function = lambda states, actions: np.array(map(lambda s, a: \
                mdp.computeQFunction(s, a, K, sigma ** 2), states, actions))
        plot_state_action_function(reward_function, 'Reward')
        plot_state_action_function(Q_function, 'Q-function')

    #Collect samples
    n_episodes = 20
    dataset = evaluation.collect_episodes(mdp, policy, n_episodes)

    estimator = ContinuousEnvSampleEstimator(dataset, mdp.gamma)
    ml_estimator = MaximumLikelihoodEstimator(dataset)
    d_sa_mu_hat = estimator.get_d_sa_mu()
    D_hat = np.diag(d_sa_mu_hat)

    states_actions = dataset[:, :2]
    states = dataset[:, 0]
    actions = dataset[:, 1]
    rewards = dataset[:, 2]
    q_function = np.array(map(lambda s, a: mdp.computeQFunction(s, a, K, sigma ** 2), states, actions))

    G = np.array(map(lambda s,a: policy.gradient_log(s,a), states, actions))
    H = np.array(map(lambda s,a: policy.hessian_log(s,a), states, actions))

    print('-' * 100)
    print('Computing Q-function approx space...')

    X = np.dot(G.T, D_hat)
    phi = la2.nullspace(X)

    psi = phi
    r_norm = rewards[:, np.newaxis] / la.norm(rewards)
    q_function_norm = q_function[:, np.newaxis] / la.norm(q_function)

    print('Computing gradients and hessians...')
    gradient_hat = ml_estimator.estimate_gradient(psi, G, use_baseline=True)
    hessian_hat = ml_estimator.estimate_hessian(psi, G, H, use_baseline=True)
    hessian_reward = ml_estimator.estimate_hessian(r_norm, G, H, use_baseline=True).ravel()
    hessian_q = ml_estimator.estimate_hessian(q_function_norm, G, H, use_baseline=True).ravel()

    '''
    Since the dimension of the parameter space is 1 if the trace and max eig are
    the same, if the hessian is positive just change the sign of the feature.
    We can always get all negative definite hessians. Thus all the features
    can be used.
    '''
    mask = (hessian_hat > 0).ravel()
    psi[:, mask] *= -1
    hessian_hat[mask] *= -1

    print('Reward hessian %s' % hessian_reward)
    print('Q-function hessian %s' % hessian_reward)
    print('Minimum hessian %s' % hessian_hat.max())
    print('Maximum hessian %s' % hessian_hat.min())

    '''
    Trace minimization, max eig minimization or hessian heuristic in this case are
    the same since the dimension of parameter space is 1
    '''

    if not fast_heuristic:
        print('Heuristic with increasing number of features ranked by trace')
        ind = hessian_hat.ravel().argsort()
        hessian_hat = hessian_hat[ind]
        psi = psi[:, ind]

        n_features = np.arange(0, 2001, 50)
        n_features[0] = 1
        n_features[-1] = psi.shape[1]
        traces = []
        for i in n_features:
            used_hessians = hessian_hat[:i]
            optimizer = HeuristicOptimizerNegativeDefinite(used_hessians)
            w = optimizer.fit()
            trace = np.tensordot(w, used_hessians, axes=1).ravel()
            traces.append(trace)

        fig, ax = plt.subplots()
        ax.set_xlabel('number of features (sorted by trace)')
        ax.set_ylabel('trace = max eigval')
        ax.plot(n_features, traces, marker='o', label='Features')
        ax.plot([1, n_features[-1]], [hessian_reward] * 2, color='r', \
                linewidth=2.0, label='Reward function')
        ax.plot([1, n_features[-1]], [hessian_hat.min()] * 2, color='g', \
                linewidth=2.0, label='Best individual feature')
        ax.plot([1, n_features[-1]], [hessian_hat.max()] * 2, color='m', \
                linewidth=2.0, label='Worst individual feature')
        ax.legend(loc='upper right')
        plt.yscale('symlog')
    else:
        optimizer = HeuristicOptimizerNegativeDefinite(hessian_hat)
        w = optimizer.fit()
        best = np.dot(psi, w)[:, np.newaxis]
        print('Best basis function %s' % np.tensordot(w, hessian_hat, axes=1).ravel())

    '''
    Reward functions are compared on the basis of the training time
    '''

    #Rescale rewards into [0,1]
    scaler = MinMaxScaler(copy=False)
    r_norm = scaler.fit_transform(r_norm)
    q_function_norm = scaler.fit_transform(q_function_norm)
    best = scaler.fit_transform(best)

    print('-' * 100)
    print('Estimating d(s,a)...')

    sigma = 4.
    def gaussian_kernel(x):
        return 1. / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(- 1. / 2 * x ** 2 / sigma ** 2)

    d_sa_knn = KNeighborsRegressor2(n_neighbors=5, weights=gaussian_kernel)
    d_sa_knn.fit(states_actions, np.ones(len(states_actions)))
    plot_state_action_function(get_knn_function_for_plot(d_sa_knn, True), 'd(s,a)')

    print('-' * 100)
    print('Training with REINFORCE')

    learner = PolicyGradientLearner(mdp, policy, lrate=0.01, verbose=1)

    #Basis functions with and without penalization

    if plot_train:
        fig, ax = plt.subplots()
        ax.set_xlabel('itearations')
        ax.set_ylabel('parameter')
        fig.suptitle('REINFORCE')

    k_list = [1, 2, 5, 10]
    penalty_list = [True, False]
    histories = []
    labels = []
    for k in k_list:
        for penalty in penalty_list:
            label = 'Trace minimizer - %sknn%s' % (k, ' - penalty' if penalty else '')
            print(label)
            history = train(learner, states_actions, gaussian_kernel, k, penalty, best, d_sa_knn)
            histories.append(history)
            labels.append(label)
            if plot_train:
                ax.plot(np.arange(len(history)), np.array(history).ravel(),
                        marker='+', label=label)


    # history_reward =
    #hystory_q_function =
    #hystory_a_function =

    print('-' * 100)
    print('Estimating maximum entropy reward...')
    from reward_space.inverse_reinforcement_learning.maximum_entropy_irl import MaximumEntropyIRL
    me = MaximumEntropyIRL(dataset)
    w = me.fit(G, phi)
    best_me = np.dot(phi, w)[:, np.newaxis]
    best_me = scaler.fit_transform(best_me)
    hystory_max_entropy = train(learner, states_actions, gaussian_kernel, 2, False, best_me, plot=True)
    if plot_train:
        ax.plot(np.arange(len(hystory_max_entropy)), np.array(hystory_max_entropy).ravel(),
                marker='+', label='Max entropy')

    print('-' * 100)
    print('Estimating proto value functions...')
    cpvf = ContinuousProtoValueFunctions(k_neighbors=5, kernel=gaussian_kernel)
    cpvf.fit(dataset)
    k_list = [1, 2, 5, 10, 20, 50, 100]
    hystories_pvf = []
    for k in k_list:
        _, phi_pvf = cpvf.transform(k)
        phi_pvf = phi_pvf.sum(axis=1)
        phi_pvf = scaler.fit_transform(phi_pvf)
        hystory = train(learner, states_actions, gaussian_kernel, 5,
                                    False, phi_pvf, plot=True)
        hystories_pvf.append(hystory)
        label = 'PVF - %sk' % k
        if plot_train:
            ax.plot(np.arange(len(history)), np.array(history).ravel(),
                    marker='+', label=label)


    if plot_train:
        ax.legend(loc='lower right')

    '''
    histories = []
    #1knn
    knn = KNeighborsRegressor(n_neighbors=1, weights=gaussian_kernel)
    knn.fit(states_actions, best.ravel())
    plot_state_action_function(lambda states, actions: knn.predict(
        np.hstack([states[:, np.newaxis], actions[:, np.newaxis]])) *0.5 + \
        1./0.16 * np.array(map(lambda s,a: predict_d_mu_sa(states_actions, s,a), states, actions)),
        'Trace mimimizer 1knn - penalty')

    theta, history1 = learner.optimize(-0.2,
        reward=lambda traj: knn.predict(traj[:, :2]) * 0.5 + \
        1./0.16 * np.array(map(lambda x: predict_d_mu_sa(states_actions, x[0],x[1], 5), traj[:, :2].tolist())),
        return_history=True)

    histories.append(np.array(history1).ravel())

    plot_state_action_function(lambda states, actions: knn.predict(
        np.hstack([states[:, np.newaxis], actions[:, np.newaxis]])),
        'Trace mimimizer 1knn - no penalty')

    theta, history1 = learner.optimize(-1.,
        reward=lambda traj: knn.predict(traj[:, :2]), return_history=True)
    histories.append(np.array(history1).ravel())


    # 2knn
    knn = KNeighborsRegressor(n_neighbors=2, weights=gaussian_kernel)
    knn.fit(states_actions, best.ravel())
    plot_state_action_function(lambda states, actions: knn.predict(
        np.hstack([states[:, np.newaxis], actions[:, np.newaxis]])) * 0.5 + \
        1. / 0.16 * np.array(map(lambda s, a: predict_d_mu_sa(states_actions, s, a), states, actions)),
        'Trace mimimizer 2knn - penalty')

    theta, history1 = learner.optimize(-0.2,
        reward=lambda traj: knn.predict(traj[:, :2]) * 0.5 + \
        1. / 0.16 * np.array(map(lambda x: predict_d_mu_sa(states_actions, x[0], x[1], 5), traj[:, :2].tolist())),
        return_history=True)
    histories.append(np.array(history1).ravel())

    plot_state_action_function(lambda states, actions: knn.predict(
        np.hstack([states[:, np.newaxis], actions[:, np.newaxis]])),
                               'Trace mimimizer 2knn - no penalty')

    theta, history1 = learner.optimize(-0.2,
        reward=lambda traj: knn.predict(traj[:, :2]), return_history=True)
    histories.append(np.array(history1).ravel())

    #5knn
    knn = KNeighborsRegressor(n_neighbors=5, weights=gaussian_kernel)
    knn.fit(states_actions, best.ravel())
    plot_state_action_function(lambda states, actions: knn.predict(
        np.hstack([states[:, np.newaxis], actions[:, np.newaxis]])) * 0.5 + \
        1. / 0.16 * np.array(map(lambda s, a: predict_d_mu_sa(states_actions, s, a), states, actions)),
        'Trace mimimizer 5knn - penalty')

    theta, history1 = learner.optimize(-0.2,
        reward=lambda traj: knn.predict(traj[:, :2]) * 0.5 + \
        1. / 0.16 * np.array(map(lambda x: predict_d_mu_sa(states_actions, x[0], x[1], 5), traj[:, :2].tolist())),
        return_history=True)
    histories.append(np.array(history1).ravel())

    plot_state_action_function(lambda states, actions: knn.predict(
        np.hstack([states[:, np.newaxis], actions[:, np.newaxis]])),
                               'Trace mimimizer 5knn - no penalty')

    theta, history1 = learner.optimize(-0.2,
        reward=lambda traj: knn.predict(traj[:, :2]), return_history=True)
    histories.append(np.array(history1).ravel())

    #10knn
    knn = KNeighborsRegressor(n_neighbors=10, weights=gaussian_kernel)
    knn.fit(states_actions, best.ravel())
    plot_state_action_function(lambda states, actions: knn.predict(
        np.hstack([states[:, np.newaxis], actions[:, np.newaxis]])) * 0.5 + \
        1. / 0.16 * np.array(map(lambda s, a: predict_d_mu_sa(states_actions, s, a), states, actions)),
        'Trace mimimizer 10knn - penalty')

    theta, history1 = learner.optimize(-0.2,
        reward=lambda traj: knn.predict(traj[:, :2]) * 0.5 + \
        1. / 0.16 * np.array(map(lambda x: predict_d_mu_sa(states_actions, x[0], x[1], 5), traj[:, :2].tolist())),
        return_history=True)
    histories.append(np.array(history1).ravel())

    plot_state_action_function(lambda states, actions: knn.predict(
        np.hstack([states[:, np.newaxis], actions[:, np.newaxis]])),
                               'Trace mimimizer 10knn - no penalty')

    theta, history1 = learner.optimize(-1.,
        reward=lambda traj: knn.predict(traj[:, :2]), return_history=True)
    histories.append(np.array(history1).ravel())

    #True reward
    knn = KNeighborsRegressor(n_neighbors=1, weights=gaussian_kernel)
    knn.fit(states_actions, r_norm.ravel())
    theta, history2 = learner.optimize(-0.2, reward=lambda traj: knn.predict(
        traj[:, :2]), return_history=True)
    theta, history1 = learner.optimize(-1.,
        reward=lambda traj: (traj[:, 2] -min(rewards)) / (max(rewards) - min(rewards)), return_history=True)
    histories.append(np.array(history1).ravel())
    '''

    '''
    learner = PolicyGradientLearner(mdp, policy, lrate=0.2, lrate_decay={'method' : 'inverse', 'decay' : .1}, verbose=1)
    theta, history1 = learner.optimize(-0.2, reward=lambda traj: knn.predict(
        traj[:, :2]), return_history=True)
    policy.set_parameter(theta)

    knn = KNeighborsRegressor(n_neighbors=10, weights=gaussian_kernel)
    knn.fit(states_actions, phi)

    knn.fit(states_actions, np.dot(phi, w))
    plot_state_action_function(lambda states, actions: knn.predict(
        np.hstack([states[:, np.newaxis], actions[:, np.newaxis]])),
                               'Trace mimimizer 10knn - no penality')


    #learner = PolicyGradientLearner(mdp, policy, lrate=0.15, verbose=1)
    theta, history4 = learner.optimize(-0.2, reward=lambda traj: knn.predict(
        traj[:, :2]), return_history=True)
    policy.set_parameter(theta)

    knn = KNeighborsRegressor(n_neighbors=1, weights=gaussian_kernel)
    knn.fit(states_actions, phi)
    knn.fit(states_actions, rewards / la.norm(rewards))
    plot_state_action_function(lambda states, actions: knn.predict(
        np.hstack([states[:, np.newaxis], actions[:, np.newaxis]])),
                               'Reward function')

    #learner = PolicyGradientLearner(mdp, policy, lrate=0.15, verbose=1)
    theta, history2 = learner.optimize(-0.2, reward=lambda traj: knn.predict(
        traj[:, :2]), return_history=True)
    policy.set_parameter(theta)

    learner = PolicyGradientLearner(mdp, policy, lrate=0.2 / la.norm(dataset[:, 2]),  lrate_decay={'method' : 'inverse', 'decay' : .1}, verbose=1)
    theta, history3 = learner.optimize(-0.2, return_history=True)
    policy.set_parameter(theta)

    fig, ax = plt.subplots()
    ax.set_xlabel('itearations')
    ax.set_ylabel('parameter')
    fig.suptitle('REINFORCE')
    ax.plot(np.arange(len(history1)), [np.asscalar(K)] * len(history1), linewidth=2.0, label='Optimal parameter')
    ax.plot(np.arange(len(history1)), np.array(history1).ravel(), marker='+', label='Trace minimizer 1knn - no penality')
    ax.plot(np.arange(len(history2)), np.array(history2).ravel(), marker='+', label='Reward as basis function - no penality')
    ax.plot(np.arange(len(history3)), np.array(history3).ravel(), marker='+', label='Reward')
    ax.plot(np.arange(len(history4)), np.array(history4).ravel(), marker='+',
            label='Trace minimizer 10knn - no penality')
    ax.legend(loc='lower right')



    fig, ax = plt.subplots()
    ax.set_xlabel('trace')
    ax.set_ylabel('max eigval')
    fig.suptitle('Hessians')
    ax.scatter(hessian_hat.ravel(), hessian_hat.ravel(), color='b', marker='+',
               label='Estimated hessians')
    ax.scatter(hessian_reward.ravel(), hessian_reward.ravel(), color='r', marker='o', label='Reward function')
    ax.scatter(traces[-1], traces[-1], color='y', marker='o',
               label='Best')
    ax.legend(loc='upper left')
    '''