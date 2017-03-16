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

def LSestimate(X, Q_true):
    '''
    Performs LS estimation of the Q function starting from the orthonormal
    basis X and the target Q_true
    '''
    w, residuals, rank, _ =  la.lstsq(X, Q_true)
    rmse = np.sqrt(residuals/X.shape[0])
    Q_hat = X.dot(w)
    return Q_hat, w, rmse

def PVF_estimation(estimator, Q_true, mu, J_true, PI, plot=False):

    #PVF basis on policy
    eigval_on, PVF_on = estimator.compute_PVF(3000, operator='norm-laplacian',
                                        method='on-policy')

    #PVF basis off policy
    eigval_off, PVF_off = estimator.compute_PVF(3000, operator='norm-laplacian',
                                        method='off-policy')
    ks = np.arange(0, 501, 50)
    ks[0] = 1

    #Polynomial basis
    poly = np.ones(3000) * np.arange(1, 3001)[:, np.newaxis] ** np.arange(0,3000)

    err_PVF_on, err_PVF_off, err_poly = [], [], []

    for k in ks:
        Q_hat, w, rmse, _ = la2.lsq(PVF_on[:, :k], Q_true)
        J_hat = la.multi_dot([mu.T, PI, Q_hat])
        err_PVF_on.append(abs(J_hat - J_true))
        print('PVF on-policy k = %s deltaJ = %s J_hat = %s rmse = %s' % (
        k, abs(J_true - J_hat), J_hat, rmse))

        Q_hat, w, rmse, _ = la2.lsq(PVF_off[:, :k], Q_true)
        J_hat = la.multi_dot([mu.T, PI, Q_hat])
        err_PVF_off.append(abs(J_hat - J_true))
        print('PVF off-policy k = %s deltaJ = %s J_hat = %s rmse = %s' % (
            k, abs(J_true - J_hat), J_hat, rmse))

        Q_hat, w, rmse, _ = la2.lsq(poly[:, :k], Q_true)
        J_hat = la.multi_dot([mu.T, PI, Q_hat])
        err_poly.append(abs(J_hat - J_true))
        print('poly k = %s deltaJ = %s J_hat = %s rmse = %s' % (
            k, abs(J_true - J_hat), J_hat, rmse))

    if plot:
        fig, ax = plt.subplots()
        ax.plot(ks, err_PVF_on, c='r',  label='norm-lapl PVF on-policy')
        ax.plot(ks, err_PVF_off, c='b', label='norm-lapl PVF off-policy')
        ax.plot(ks, err_poly, c='g',label='polynomial basis')
        ax.legend(loc='upper right')
        ax.set_xlabel('number of features')
        ax.set_ylabel('delta J')
        plt.savefig('img/Q-function approx.png')
        plt.show()

    return (eigval_on, PVF_on), (eigval_off, PVF_off)

def Q_estimation(C, d_sa_mu_hat, Q_true, mu, PI, J_true, PVF, eigval, mdp_wrap, estimator, PI_tilde):

    d_sa_mu = mdp_wrap.compute_d_sa_mu()
    A_true = (np.eye(mdp_wrap.nA * mdp_wrap.nS) - PI_tilde).dot(Q_true)

    #Find the orthogonal complement
    Phi_Q = la2.nullspace(np.dot(C.T, np.diag(d_sa_mu_hat)))
    print('Number of Q-features (rank of Phi_Q) %s' % la.matrix_rank(Phi_Q))

    Q_hat, w, rmse, _ = la2.lsq(Phi_Q, Q_true)
    J_hat = la.multi_dot([mu.T, PI, Q_hat])
    print('Results of LS deltaJ = %s J_hat = %s rmse = %s' % (
            abs(J_true - J_hat), J_hat, rmse))

    #Project PVFs onto orthogonal complement
    PVF_hat, W, _, _ = la2.lsq(Phi_Q, PVF)

    #New basis function rank
    rank = eigval / la.norm(PVF_hat, axis=0)
    ind = rank.argsort()
    PVF_hat = PVF_hat[:, ind]

    PVF_hat = PVF_hat / la.norm(PVF_hat, axis=0)

    ks = np.arange(0, 501, 50)
    ks[0] = 1

    err_PVF_hat, err_rand, sdev_err_rand = [], [], []
    err_PVF_hat_A, err_rand_A, sdev_err_rand_A = [], [], []
    trials = 20

    norm_Q_true = np.asscalar(la.multi_dot([Q_true, np.diag(d_sa_mu), Q_true[:, np.newaxis]]))
    norm_A_true = np.asscalar(la.multi_dot([A_true, np.diag(d_sa_mu), A_true[:, np.newaxis]]))

    for k in ks:
        Q_hat, w, rmse, _ = la2.lsq(PVF_hat[:, :k], Q_true)
        #A_hat = (np.eye(mdp_wrap.nA * mdp_wrap.nS) - PI_tilde).dot(Q_hat)
        A_hat, _, _, _ = la2.lsq((np.eye(mdp_wrap.nA * mdp_wrap.nS) - PI_tilde).dot(PVF_hat[:, :k]), A_true)

        J_hat = la.multi_dot([mu.T, PI, Q_hat])
        #err_PVF_hat.append(abs(J_hat - J_true))
        print('PVF_hat k = %s deltaJ = %s J_hat = %s rmse = %s' % (
        k, abs(J_true - J_hat), J_hat, rmse))

        err_PVF_hat_A.append(np.asscalar(la.multi_dot([(A_true-A_hat), np.diag(d_sa_mu), (A_true-A_hat)[:, np.newaxis]])) / norm_A_true)
        err_PVF_hat.append(np.asscalar(la.multi_dot([(Q_true-Q_hat), np.diag(d_sa_mu), (Q_true-Q_hat)[:, np.newaxis]])) / norm_Q_true)

        deltaJ = []
        rmsem = 0

        err_ite_Q, err_ite_A = [], []
        for i in range(trials):
            choice = np.random.choice(PVF_hat.shape[1], k, replace=False)
            Q_hat, w, rmse, _ = la2.lsq(PVF_hat[:,choice], Q_true)
            A_hat, _, _, _ = la2.lsq(
                (np.eye(mdp_wrap.nA * mdp_wrap.nS) - PI_tilde).dot(
                    PVF_hat[:, choice]), A_true)

            J_hat = la.multi_dot([mu.T, PI, Q_hat])
            deltaJ.append(abs(J_hat - J_true))
            rmsem += rmse

            err_ite_A.append(np.asscalar(
                la.multi_dot([(A_true - A_hat), np.diag(d_sa_mu),
                             (A_true - A_hat)[:, np.newaxis]])) / norm_A_true)
            err_ite_Q.append(np.asscalar(
                la.multi_dot([(Q_true - Q_hat), np.diag(d_sa_mu),
                             (Q_true - Q_hat)[:, np.newaxis]])) / norm_Q_true)

        err_rand.append(np.mean(err_ite_Q))
        sdev_err_rand.append(np.std(err_ite_Q))

        err_rand_A.append(np.mean(err_ite_A))
        sdev_err_rand_A.append(np.std(err_ite_A))

        #err_rand.append(np.mean(deltaJ))
        #sdev_err_rand.append(np.std(deltaJ))
        print('random k = %s deltaJ = %s J_hat = %s rmse = %s' % (
                k, np.mean(deltaJ), J_hat, rmsem/trials))

    fig, ax = plt.subplots()
    ax.plot(ks, err_PVF_hat, marker='o', label='gradient PVF')
    ax.errorbar(ks, err_rand, marker='d', yerr=sdev_err_rand, label='random')
    ax.legend(loc='upper right')
    ax.set_xlabel('number of features')
    #ax.set_ylabel('delta J')
    ax.set_ylabel('||Q_hat-Q_true||d(s,a) / ||Q_true||d(s,a)')
    fig.suptitle('Q-function approx with PVF')
    plt.savefig('img/Q-function approx gradient.png')

    fig, ax = plt.subplots()
    ax.plot(ks, err_PVF_hat_A, marker='o', label='gradient PVF')
    ax.errorbar(ks, err_rand_A, marker='d', yerr=sdev_err_rand_A, label='random')
    ax.legend(loc='upper right')
    ax.set_xlabel('number of features')
    #ax.set_ylabel('delta J')
    ax.set_ylabel('||A_hat-A_true||d(s,a) / ||A_true||d(s,a)')
    fig.suptitle('A-function approx with PVF')
    plt.savefig('img/A-function approx gradient.png')
    plt.show()

    return Phi_Q, PVF_hat

def R_estimation(estimator, mdp_wrap, PI, Phi_Q, d_sa_mu, J_true, R, PVF_hat):
    P_hat = estimator.get_P()
    Phi_R_1 = np.dot(np.eye(mdp_wrap.nA * mdp_wrap.nS) - mdp_wrap.gamma * np.dot(P_hat, PI),Phi_Q)
    print('Number of R-features (rank of Phi_R_1) %s' % la.matrix_rank(Phi_R_1))

    print(la.matrix_rank(PVF_hat))

    PRF_hat = np.dot(np.eye(mdp_wrap.nA * mdp_wrap.nS) - mdp_wrap.gamma * np.dot(estimator.P, PI),PVF_hat)

    PRF_hat = PRF_hat / la.norm(PRF_hat, axis=0)

    print(la.matrix_rank(PRF_hat))
    print(la.matrix_rank(Phi_R_1))
    print(la.matrix_rank(np.hstack([Phi_R_1, PRF_hat])))

    R_hat_1, w, rmse, _ = la2.lsq(PRF_hat, R)
    J_hat = np.dot(d_sa_mu, R_hat_1)
    print('Results of LS deltaJ = %s J_hat = %s rmse = %s' % (
        abs(J_true - J_hat), J_hat, rmse))

    ks = np.arange(0, 501, 50)
    ks[0] = 1

    err_PVF_hat, err_rand, sdev_err_rand = [], [], []
    trials = 20
    for k in ks:
        R_hat, w, rmse, _ = la2.lsq(PRF_hat[:, :k], R)

        J_hat = np.dot(d_sa_mu, R_hat)
        err_PVF_hat.append(abs(J_hat - J_true))
        print('PRF_hat k = %s deltaJ = %s J_hat = %s rmse = %s' % (
            k, abs(J_true - J_hat), J_hat, rmse))

        deltaJ = []
        rmsem = 0
        deltaJ_2 = 0
        for i in range(trials):
            choice = np.random.choice(Phi_R_1.shape[1], k, replace=False)
            R_hat, w, rmse, _ = la2.lsq(Phi_R_1[:, choice], R)

            J_hat = np.dot(d_sa_mu, R_hat)
            deltaJ.append(abs(J_hat - J_true))
            rmsem += rmse

        err_rand.append(np.mean(deltaJ))
        sdev_err_rand.append(np.std(deltaJ))
        print('random k = %s deltaJ = %s J_hat = %s rmse = %s' % (
            k, np.mean(deltaJ), J_hat, rmsem / trials))

    fig, ax = plt.subplots()
    ax.plot(ks, err_PVF_hat, marker='o', label='gradient PVF')
    ax.errorbar(ks, err_rand, marker='d', yerr=sdev_err_rand, label='random')
    ax.legend(loc='upper right')
    ax.set_xlabel('number of features')
    ax.set_ylabel('delta J')
    plt.savefig('img/R-function approx 1.png')
    plt.show()

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

def compute(mdp, n_episodes, prox_policy, opt_policy, random_policy, plot=False):
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
        plot_state_function(mdp, mdp_wrap.compute_V_function()[:500], 'V-function')
        plot_state_action_function(mdp, abs(R - A_true) / abs(R+1e-24) * mdp_wrap.compute_d_sa_mu(), 'relative error R vs A')
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

    PI_tilde = np.repeat(PI, mdp_wrap.nA, axis=0)
    A_true = (np.eye(mdp_wrap.nA * mdp_wrap.nS) - PI_tilde).dot(Q_true)

    Phi_A = (np.eye(mdp_wrap.nA * mdp_wrap.nS) - PI_tilde).dot(Phi_Q)
    print('Number of R-features (rank of Phi_R) %s' % la.matrix_rank(Phi_A))
    Phi_A = la2.range(Phi_A)

    A_hat, w, rmse, _ = la2.lsq(Phi_A, A_true)

    if plot:
        plot_state_action_function(mdp, Q_hat, 'Q-function approx')
        plot_state_action_function(mdp, A_hat, 'A-function approx')
        plot_state_action_function(mdp, abs(A_hat - A_true) / abs(A_true+1e-24) * mdp_wrap.compute_d_sa_mu(), 'relative error A vs A_hat')
        plot_state_action_function(mdp, abs(Q_hat - Q_true) / abs(Q_true+1e-24) * mdp_wrap.compute_d_sa_mu(), 'relative error Q vs Q_hat')

    print('-' * 100)
    (eigval, pvf) , (_, _) = PVF_estimation(estimator, Q_true, mu, J_true, PI, plot=False)

    print('-' * 100)
    Q_estimation(C, d_sa_mu_hat, Q_true, mu, PI, J_true, pvf, eigval, mdp_wrap, estimator, PI_tilde)

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
prox_policy = TaxiEnvPolicy2StateParameter(sigma=0.02)
compute(mdp, n_episodes, prox_policy, opt_policy, random_policy)

