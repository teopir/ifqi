from __future__ import print_function
from ifqi.envs import TaxiEnv
from policy import  TaxiEnvPolicy, TaxiEnvPolicyStateParameter, TaxiEnvPolicy2StateParameter, TaxiEnvPolicyOneParameter, TaxiEnvPolicy2Parameter, TaxiEnvRandomPolicy
from ifqi.evaluation import evaluation
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from reward_space.utils.discrete_env_sample_estimator import DiscreteEnvSampleEstimator
from reward_space.utils.discrete_mdp_wrapper import DiscreteMdpWrapper
import reward_space.utils.linalg2 as la2

def mdp_norm(f, mdp_wrap):
    d_sa_mu = mdp_wrap.compute_d_sa_mu()
    res = la.multi_dot([f, np.diag(d_sa_mu), f[:, np.newaxis]])
    return np.sqrt(res / sum(d_sa_mu))

def pvf_estimation(estimator, mdp_wrap, perform_estimation=False, plot=False):

    #PVF basis on policy
    eigval_on, pvf_on = estimator.compute_PVF(mdp_wrap.nS * mdp_wrap.nA, method='on-policy')
    #PVF basis off policy
    eigval_off, pvf_off = estimator.compute_PVF(mdp_wrap.nS * mdp_wrap.nA, method='off-policy')

    if not perform_estimation:
        return (eigval_on, pvf_on), (eigval_off, pvf_off)

    ks = np.arange(0, 51, 5)
    ks[0] = 1

    #Polynomial basis
    poly = np.ones(mdp_wrap.nS * mdp_wrap.nA) * np.arange(1, mdp_wrap.nS * \
        mdp_wrap.nA + 1)[:, np.newaxis] ** np.arange(0,mdp_wrap.nS * mdp_wrap.nA)

    err_PVF_on, err_PVF_off, err_poly = [], [], []
    Q_true = mdp_wrap.compute_Q_function()
    norm_Q = mdp_norm(Q_true, mdp_wrap)

    for k in ks:
        Q_hat, w, rmse, _ = la2.lsq(pvf_on[:, :k], Q_true)
        err_PVF_on.append(mdp_norm(Q_true - Q_hat, mdp_wrap) / norm_Q)
        print('PVF on-policy k = %s error = %s' % (k, err_PVF_on[-1]))

        Q_hat, w, rmse, _ = la2.lsq(pvf_off[:, :k], Q_true)
        err_PVF_off.append(mdp_norm(Q_true - Q_hat, mdp_wrap) / norm_Q)
        print('PVF off-policy k = %s error = %s' % (k, err_PVF_off[-1]))

        Q_hat, w, rmse, _ = la2.lsq(poly[:, :k], Q_true)
        err_poly.append(mdp_norm(Q_true - Q_hat, mdp_wrap) / norm_Q)
        print('poly k = %s error = %s' % (k, err_poly[-1]))

    if plot:
        fig, ax = plt.subplots()
        ax.plot(ks, err_PVF_on, c='r', marker='o', label='norm-lapl PVF on-policy')
        ax.plot(ks, err_PVF_off, c='b', marker='o', label='norm-lapl PVF off-policy')
        ax.plot(ks, err_poly, c='g', marker='o', label='polynomial basis')
        ax.legend(loc='upper right')
        ax.set_xlabel('number of features')
        ax.set_ylabel('||Q_hat-Q_true||d(s,a) / ||Q_true||d(s,a)')
        plt.savefig('img/Q-function approx.png')
        plt.show()

    return (eigval_on, pvf_on), (eigval_off, pvf_off)

def Q_estimation(C, d_sa_mu_hat, Q_true, mu, PI, J_true, PVF, eigval, mdp_wrap, estimator, PI_tilde):

    d_sa_mu = mdp_wrap.compute_d_sa_mu()
    Q_true = mdp_wrap.compute_Q_function()
    V_true = np.repeat(mdp_wrap.compute_V_function()[:mdp_wrap.nS], mdp_wrap.nA)
    A_true = Q_true - V_true

    norm_Q = mdp_norm(Q_true, mdp_wrap)
    norm_A = mdp_norm(A_true, mdp_wrap)

    #Find the orthogonal complement
    Phi_Q = la2.nullspace(np.dot(C.T, np.diag(d_sa_mu_hat)))
    print('Number of Q-features (rank of Phi_Q) %s' % la.matrix_rank(Phi_Q))

    #Project PVFs onto orthogonal complement
    PVF_hat, W, _, _ = la2.lsq(Phi_Q, PVF)

    #New basis function rank
    rank = eigval / la.norm(PVF_hat, axis=0)
    ind = rank.argsort()
    rank = rank[ind]
    PVF_hat = PVF_hat[:, ind]

    PVF_hat = PVF_hat / la.norm(PVF_hat, axis=0)

    ks = np.arange(0, 51, 5)
    ks[0] = 1

    err_PVF_hat, err_rand, sdev_err_rand = [], [], []
    err_PVF_hat_A, err_rand_A, sdev_err_rand_A = [], [], []

    trials = 20

    PRF_hat = np.dot(np.eye(mdp_wrap.nS * mdp_wrap.nA) - PI_tilde, PVF_hat)
    print(rank)
    rank = rank / la.norm(PRF_hat, axis=0)
    print(rank)
    ind = rank.argsort()
    PRF_hat = PRF_hat[:, ind]

    for k in ks:
        PVF_hat_k = la2.range(PVF_hat[:, :k])
        Q_hat, w, rmse, _ = la2.lsq(PVF_hat_k, Q_true)
        PRF_hat_k = la2.range(PRF_hat[:, :k])
        A_hat, _, _, _ = la2.lsq(PRF_hat_k, A_true)

        err_PVF_hat_A.append(mdp_norm(A_true-A_hat, mdp_wrap) / norm_A)
        err_PVF_hat.append(mdp_norm(Q_true-Q_hat, mdp_wrap) / norm_Q)

        print('GPVF ranked k = %s error = %s' % (k, err_PVF_hat[-1]))
        print('GPRF ranked k = %s error = %s' % (k, err_PVF_hat_A[-1]))

        err_ite_Q, err_ite_A = [], []
        for i in range(trials):
            choice = np.random.choice(PVF_hat.shape[1], k, replace=False)
            Q_hat, w, rmse, _ = la2.lsq(PVF_hat[:,choice], Q_true)
            V_hat = np.repeat(PI.dot(Q_hat), mdp_wrap.nA)
            A_hat = Q_hat - V_hat

            err_ite_A.append(mdp_norm(A_true-A_hat, mdp_wrap) / norm_A)
            err_ite_Q.append(mdp_norm(Q_true-Q_hat, mdp_wrap) / norm_Q)

        err_rand.append(np.mean(err_ite_Q))
        sdev_err_rand.append(np.std(err_ite_Q))

        err_rand_A.append(np.mean(err_ite_A))
        sdev_err_rand_A.append(np.std(err_ite_A))

        print('GPVF random k = %s error = %s' % (k, err_rand[-1]))
        print('GPRF random k = %s error = %s' % (k, err_rand_A[-1]))

    fig, ax = plt.subplots()
    ax.plot(ks, err_PVF_hat, marker='o', label='GPVF ranked')
    ax.errorbar(ks, err_rand, marker='d', yerr=sdev_err_rand, label='GPVF random')
    ax.legend(loc='upper right')
    ax.set_xlabel('number of features')
    ax.set_ylabel('||Q_hat-Q_true||d(s,a) / ||Q_true||d(s,a)')
    fig.suptitle('Q-function approx with PVF')
    plt.savefig('img/Q-function approx gradient.png')

    fig, ax = plt.subplots()
    ax.plot(ks, err_PVF_hat_A, marker='o', label='GPRF ranked')
    ax.errorbar(ks, err_rand_A, marker='d', yerr=sdev_err_rand_A, label='GPRF random')
    ax.legend(loc='upper right')
    ax.set_xlabel('number of features')
    ax.set_ylabel('||A_hat-A_true||d(s,a) / ||A_true||d(s,a)')
    fig.suptitle('A-function approx with PVF')
    plt.savefig('img/A-function approx gradient.png')
    plt.show()

    return Phi_Q, PVF_hat


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


def true_PVF(P):
    PI_random = np.repeat(np.eye(500),6, axis=1) / 6.
    W = np.dot(PI_random, P)

    W = .5 * (W + W.T)
    W[W.nonzero()] = 1

    d = W.sum(axis=1)
    D = np.diag(d)
    D1 = np.diag(np.power(d, -0.5))

    #L = la.multi_dot([D1, D - W, D1])
    L=D-W
    print(np.dot(D - W,np.ones((500,1))))

    eigval, eigvec = la.eigh(L)

    for i in range(5):
        plot_state_function(mdp, eigvec[:,i] , 'PVF-function')


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
    n_states_actions = policy_hessian.shape[0]
    n_params = policy_hessian.shape[1]
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
        episode_hessian[episode, :, :] += np.outer(policy_gradient[index, :].squeeze(), \
                    policy_gradient[index, :].squeeze()) + policy_hessian[index, :, :].squeeze()

        if dataset[i, -1] == 1:
            episode += 1

        i += 1

    return_hessians = 1. / n_episodes * np.tensordot(episode_reward_features.T, episode_hessian, axes=1)


    import scipy.optimize as opt

    def loss(x):
        hessian = np.tensordot(x, return_hessians, axes = 1)
        eigval, eigvec = la.eigh(hessian)
        return eigval[-1]

    def constraint(x):
        return la.norm(np.dot(reward_features, x)) - 1

    res = opt.minimize(loss, np.ones(n_features),
                       constraints=({'type': 'eq', 'fun': constraint}),
                       options={'disp': True},
                       tol=1e-24)
    print(res)

    return res.x

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




def compute(mdp, n_episodes, prox_policy, opt_policy, plot=False):
    print('Collecting samples from optimal approx policy...')
    dataset = evaluation.collect_episodes(mdp, prox_policy, n_episodes)
    print('Dataset made of %s samples' % (dataset.shape[0]))

    mdp_wrap = DiscreteMdpWrapper(mdp, episodic=True)

    PI_opt = opt_policy.get_distribution()
    PI = prox_policy.get_distribution()
    C = prox_policy.gradient_log_pdf()

    estimator = DiscreteEnvSampleEstimator(dataset,
                                mdp_wrap.gamma,
                                mdp_wrap.state_space,
                                mdp_wrap.action_space)

    #Optimal deterministic policy
    mdp_wrap.set_policy(PI_opt)
    J_opt = mdp_wrap.compute_J()

    #Optimal approx policy
    mdp_wrap.set_policy(PI)
    mu = mdp_wrap.non_ep_mu
    d_sa_mu = mdp_wrap.compute_d_sa_mu()
    R = mdp_wrap.non_ep_R
    J_true = mdp_wrap.compute_J()
    Q_true = mdp_wrap.compute_Q_function()

    PI_tilde = np.repeat(PI, mdp_wrap.nA, axis=0)
    A_true = (np.eye(mdp_wrap.nA * mdp_wrap.nS) - PI_tilde).dot(Q_true)

    if plot:
        plot_state_action_function(mdp, Q_true, 'Q-function')
        plot_state_action_function(mdp, R , 'R-function')
        plot_state_action_function(mdp, A_true, 'A-function')
        plot_state_action_function(mdp, mdp_wrap.compute_d_sa_mu(), 'd(s,a)')
        plot_state_function(mdp, mdp_wrap.compute_V_function()[:mdp_wrap.nS], 'V-function')
        plt.show()

    #Sample estimates
    d_s_mu_hat = estimator.get_d_s_mu()
    d_sa_mu_hat = np.dot(PI.T, d_s_mu_hat)
    J_hat = estimator.get_J()

    print('-' * 100)
    print('Expected reward opt det policy J = %g' % J_opt)
    print('True expected reward approx opt policy J_true = %g' % J_true)
    print('Estimated expected reward approx opt policy J_true = %g' % J_hat)

    grad_J_hat = 1.0 / n_episodes * la.multi_dot([C.T, np.diag(d_sa_mu_hat), Q_true])
    grad_J_true = la.multi_dot([C.T, np.diag(d_sa_mu), Q_true])
    print('Dimension of the subspace %s' % la.matrix_rank(np.dot(C.T, np.diag(d_sa_mu_hat))))
    print('True policy gradient (2-norm) %s' % la.norm(grad_J_hat))
    print('Estimated policy gradient (2-norm)%s' % la.norm(grad_J_true))

    Phi_Q = la2.nullspace(np.dot(C.T, np.diag(d_sa_mu_hat)))
    Q_hat, w, rmse, _ = la2.lsq(Phi_Q, Q_true)
    print('Number of Q-features (rank of Phi_Q) %s' % la.matrix_rank(Phi_Q))

    Phi_A = (np.eye(mdp_wrap.nA * mdp_wrap.nS) - PI_tilde).dot(Phi_Q)
    Phi_A = la2.range(Phi_A)
    print('Number of R-features (rank of Phi_R) %s' % la.matrix_rank(Phi_A))
    A_hat, w, rmse, _ = la2.lsq(Phi_A, A_true)

    if plot:
        plot_state_action_function(mdp, Q_hat, 'Q-function approx')
        plot_state_action_function(mdp, A_hat, 'A-function approx')
        plot_state_action_function(mdp, abs(A_hat - A_true) / abs(A_true+1e-24) * mdp_wrap.compute_d_sa_mu(), 'relative error A vs A_hat')
        plot_state_action_function(mdp, abs(Q_hat - Q_true) / abs(Q_true+1e-24) * mdp_wrap.compute_d_sa_mu(), 'relative error Q vs Q_hat')

    print('-' * 100)
    (eigval_on, pvf_on) , (eigval_off, pvf_off) = pvf_estimation(estimator, mdp_wrap, perform_estimation=False, plot=False)

    print('-' * 100)
    _, phi = Q_estimation(C, d_sa_mu_hat, Q_true, mu, PI, J_true, pvf_on, eigval_on, mdp_wrap, estimator, PI_tilde)

    phi = (np.eye(mdp_wrap.nA * mdp_wrap.nS) - PI_tilde).dot(phi)
    G = prox_policy.gradient_log_pdf()
    H = prox_policy.H.T

    print(phi.shape)

    #w = estimate_hessian(dataset, n_episodes, G, H, phi[:,:200], mdp_wrap.state_space, mdp_wrap.action_space)
    w = maximum_entropy(dataset, n_episodes, G, phi[:,:20], mdp_wrap.state_space, mdp_wrap.action_space)
    R_hat_hessian = np.dot(phi[:,:20], w)
    print(R_hat_hessian)
    print(la.norm(R_hat_hessian))
    A_true = A_true / la.norm(A_true)
    norm = np.asscalar(la.multi_dot(
        [(A_true ), np.diag(d_sa_mu),
         (A_true )[:, np.newaxis]]))
    error = np.asscalar(la.multi_dot([(A_true - R_hat_hessian), np.diag(d_sa_mu), (A_true - R_hat_hessian)[:, np.newaxis]]))
    print(A_true)
    print(la.norm(A_true - R_hat_hessian))
    print(norm)


    '''
    phi_tau = compute_trajectory_features(dataset, Phi_A, mdp_wrap.state_space, mdp_wrap.action_space)

    def loss(w):
        num = np.exp(np.dot(phi_tau, w))
        den = np.sum(num, axis=0)
        return -np.sum(np.log(num/den))

    def gradient(w):
        pw = np.exp(np.dot(phi_tau, w))
        num = np.sum(phi_tau * pw[:,np.newaxis], axis=0)
        den = np.sum(pw, axis=0)
        return -(np.sum(phi_tau, axis=0) - num/den)


    import scipy.optimize as opt
    x0 = np.zeros(phi_tau.shape[1])
    print(opt.minimize(loss, x0, tol=1e-24))
    '''

mdp = TaxiEnv()
n_episodes = 1000

opt_policy = TaxiEnvPolicy()
random_policy = TaxiEnvRandomPolicy()

'''
print('1 GLOBAL PARAMETER')
prox_policy = TaxiEnvPolicyOneParameter(3, 0.2, 3)
compute(mdp, n_episodes, prox_policy, opt_policy)

print('\n2 GLOBAL PARAMETERS')
prox_policy = TaxiEnvPolicy2Parameter(sigma=0.02)
compute(mdp, n_episodes, prox_policy, opt_policy)

print('\n1 STATE PARAMETER')
prox_policy = TaxiEnvPolicyStateParameter(3, 0.2, 3)
compute(mdp, n_episodes, prox_policy, opt_policy)
'''
print('\n2 STATE PARAMETERS')
prox_policy = TaxiEnvPolicy2StateParameter(sigma=0.01)
compute(mdp, n_episodes, prox_policy, opt_policy)

