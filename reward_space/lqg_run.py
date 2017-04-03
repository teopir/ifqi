from __future__ import print_function
from ifqi.envs import LQG1D
from ifqi.evaluation import evaluation
import numpy as np
from numpy import linalg as LA
import ifqi.envs as envs
from policy import GaussianPolicy1D
from utils.utils import chebvalNd, remove_projections, compute_feature_matrix, find_basis
from utils.continuous_env_sample_estimator import ContinuousEnvSampleEstimator
import utils.linalg2 as la2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler


def estimate_hessian(dataset, n_episodes, policy_gradient, policy_hessian, reward_features):

    n_samples = dataset.shape[0]
    n_features = reward_features.shape[1]
    n_params = policy_hessian.shape[1]

    episode_reward_features = np.zeros((n_episodes, n_features))
    episode_hessian = np.zeros((n_episodes, n_params, n_params))

    #baseline = estimate_hessian_baseline(dataset, n_episodes, policy_gradient, policy_hessian, reward_features, state_space, action_space)

    i = 0
    episode = 0
    while i < n_samples:
        index = i

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


if __name__ == '__main__':
    mdp = LQG1D()
    #Policy parameters
    K = mdp.computeOptimalK()
    sigma = 0.01

    policy = GaussianPolicy1D(K,sigma)

    #Collect samples
    n_episodes = 20
    dataset = evaluation.collect_episodes(mdp, policy, n_episodes)
    #dataset = dataset[(np.arange(500,1000) + np.arange(0,20000,1000)[:,np.newaxis]).ravel()]

    estimator = ContinuousEnvSampleEstimator(dataset, mdp.gamma)
    d_sa_mu_hat = estimator.get_d_sa_mu()
    D_hat = np.diag(d_sa_mu_hat)

    states = dataset[:, 0]
    actions = dataset[:, 1]
    rewards = dataset[:, 2]
    G = map(lambda s,a: policy.gradient_log(s,a), states, actions)
    H = map(lambda s,a: policy.hessian_log(s,a), states, actions)
    G = np.array(G)
    H = np.array(H)


    print('-' * 100)
    print('Computing Q-function approx space...')
    X = np.dot(G.T, D_hat)
    phi = la2.nullspace(X)
    psi = phi

    scaler = MinMaxScaler(copy=False)
    psi = scaler.fit_transform(psi)
    r = scaler.fit_transform(rewards[:, np.newaxis])

    h = estimate_hessian(dataset, n_episodes, G, H, psi)
    plt.scatter(np.arange(len(h)), h)

    h = estimate_hessian(dataset, n_episodes, G, H, r)
    plt.scatter(np.arange(len(h)), h, color='r')

    '''
    states_actions = dataset[:,:2]
    discounts = dataset[:,4]
    rewards = dataset[:,2]

    print('Dataset (sigma %f) has %d samples' % (sigma, dataset.shape[0]))

    #Scale data
    bounds = [[-max_pos, max_pos], [-max_action ,max_action]]
    scaler = MinMaxScaler(ndim=2, input_ranges=bounds)
    scaled_states_actions = scaler.scale(states_actions)
    scaled_states = scaled_states_actions[:,0]
    scaled_actions = scaled_states_actions[:,1]

    #Compute feature matrix

    complement = [lambda x : policy.gradient_log_pdf(x[0],x[1])]
    max_degree = 5
    degrees = [[ds,da] for ds in range(max_degree+1) for da in range(max_degree+1)]
    cheb_basis = map(lambda d: lambda x: chebvalNd(x, d), degrees)
    n_samples = dataset.shape[0]
    n_features = len(cheb_basis)
    n_complement = 1

    X = compute_feature_matrix(n_samples, n_features, scaled_states, scaled_actions, cheb_basis)
    C = compute_feature_matrix(n_samples, n_complement, states, actions, complement)

    W = estimator.d_sasa_mu
    X_ort = remove_projections(X, C, W)
    #X_ort = remove_projections(X, C, np.diag(discounts))
    #Non mi interessa che sia ortonormale!!!
    #X_ort_ort = find_basis(X_ort, np.diag(discounts))
    #print('Rank of feature matrix X %s/%s' % (X_ort_ort.shape[1], X.shape[1]))

    rewards_hat, w, rmse, _ = la2.lsq(X_ort, rewards)
    error = np.abs(rewards - rewards_hat)
    mae = np.mean(error)
    error_rel = np.abs((rewards - rewards_hat)/rewards)
    mare = np.mean(error_rel)


    grad_J_true = 1.0/n_episodes * LA.multi_dot([C.T, W, rewards])
    grad_J_hat = 1.0/n_episodes * LA.multi_dot([C.T, W, rewards])
    J_hat = 1.0/n_episodes * np.sum(rewards * discounts)
    print('Results of LS rmse = %s mae = %s mare = %s' % (rmse, mae, mare))
    print('True policy gradient %s' % grad_J_true)
    print('Estimated policy gradient %s' % grad_J_hat)
    print('Estimated expected return %s' % J_hat)



    #---------------------------Q-function evaluation-----------------------------
    Q_true = np.array(map(lambda s,a: mdp.computeQFunction(s, a, K, np.power(sigma,2)), states, actions))
    Q_hat, w, rmse = estimate_Q(X_ort_ort, Q_true)
    error = np.abs(Q_true - Q_hat)
    mae = np.mean(error)
    error_rel = np.abs((Q_true - Q_hat)/Q_true)
    mare = np.mean(error_rel)

    W = np.diag(discounts)

    grad_J_true = 1.0/n_episodes * LA.multi_dot([C.T, W, Q_true])
    grad_J_hat = 1.0/n_episodes * LA.multi_dot([C.T, W, Q_hat])
    J_hat = 1.0/n_episodes * np.sum(rewards * discounts)
    print('Results of LS rmse = %s mae = %s mare = %s' % (rmse, mae, mare))
    print('True policy gradient %s' % grad_J_true)
    print('Estimated policy gradient %s' % grad_J_hat)
    print('Estimated expected return %s' % J_hat)

    #-------------------------Plot------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(states, actions, Q_true, c='r', marker='o')
    ax.scatter(states, actions, Q_hat, c='b', marker='^')
    ax.set_xlabel('s')
    ax.set_ylabel('a')
    ax.set_zlabel('Q(s,a)')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(states, actions, error, c='g', marker='*')
    ax.set_xlabel('s')
    ax.set_ylabel('a')
    ax.set_zlabel('error(s,a)')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(states, error, c='g', marker='*')
    ax.set_xlabel('s')
    ax.set_ylabel('|Q_true(s,*) - Q_hat(s,*)|')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(states, error_rel, c='g', marker='*')
    ax.set_xlabel('s')
    ax.set_ylabel('|Q_true(s,*) - Q_hat(s,*)|/|Q_true(s,*)|')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(states[:n_episodes], Q_true[:n_episodes], c='r', marker='o')
    ax.scatter(states[:n_episodes], Q_hat[:n_episodes], c='b', marker='^')
    ax.set_xlabel('s')
    ax.set_ylabel('Q(s,*)')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(actions[:n_episodes], Q_true[:n_episodes], c='r', marker='o')
    ax.scatter(actions[:n_episodes], Q_hat[:n_episodes], c='b', marker='^')
    ax.set_xlabel('a')
    ax.set_ylabel('Q(*,a)')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(states[:n_episodes], error_rel[:n_episodes], c='g', marker='*')
    ax.set_xlabel('s')
    ax.set_ylabel('|Q_true(s,*) - Q_hat(s,*)|/|Q_true(s,*)|')

    plt.show()
    '''