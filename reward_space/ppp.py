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


def plot_curve(mdp, reward=None):
    k_list = np.arange(-1, -0.1, 0.01)
    returns = []
    episodes = 20
    means = []
    stds = []
    for k in k_list:
        policy.set_parameter(k)
        for i in range(episodes):
            dataset = evaluation.collect_episodes(mdp, policy, 1)
            if reward is None:
                returns.append(np.dot(dataset[:, 2], dataset[:, 4]))
            else:
                returns.append(np.dot(reward[dataset[:, 0] * 6 + dataset[:, 1]], dataset[:, 4]))
        means.append(np.mean(returns))
        stds.append(np.std(returns))

    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots()
    ax.set_xlabel('k')
    ax.set_ylabel('J(k)')
    ax.plot(k_list, means, 'k', color='#3F7F4C', marker='+')
    ax.fill_between(k_list, means - stds, means + stds,
                    alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99',
                    linewidth=0)


if __name__ == '__main__':

    plot = False
    plot_train = True
    fast_heuristic = True

    mdp = LQG1D()

    #Policy parameters
    K = mdp.computeOptimalK()
    sigma = np.sqrt(0.1)

    policy = GaussianPolicy1D(K, sigma)
    plot_curve(mdp)
    plt.show()
