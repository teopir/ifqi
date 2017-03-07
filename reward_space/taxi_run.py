from __future__ import print_function
from ifqi.envs import TaxiEnv
from policy import  TaxiEnvPolicy, TaxiEnvPolicyStateParameter, TaxiEnvPolicy2StateParameter, TaxiEnvPolicyOneParameter, TaxiEnvPolicy2Parameter, TaxiEnvRandomPolicy
from ifqi.evaluation import evaluation
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
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

def PVF_estimation(estimator, Q_true, mu, J_true, PI):

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

def Q_estimation(C, d_sa_mu_hat, Q_true, mu, PI, J_true, PVF, eigval, mdp_wrap, estimator):

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
    trials = 20
    for k in ks:
        Q_hat, w, rmse, _ = la2.lsq(PVF_hat[:, :k], Q_true)

        R_hat = np.dot(np.eye(mdp_wrap.nA * mdp_wrap.nS) - mdp_wrap.gamma * np.dot(estimator.P, PI), Q_hat[:,np.newaxis])

        J_hat = la.multi_dot([mu.T, PI, Q_hat])
        err_PVF_hat.append(abs(J_hat - J_true))
        print('PVF_hat k = %s deltaJ = %s J_hat = %s rmse = %s' % (
        k, abs(J_true - J_hat), J_hat, rmse))

        J_hat = np.dot(mdp_wrap.compute_d_sa_mu(), R_hat)
        print('RRR PVF_hat k = %s deltaJ = %s J_hat = %s rmse = %s' % (
            k, abs(J_true - J_hat), J_hat, rmse))

        deltaJ = []
        rmsem = 0
        for i in range(trials):
            choice = np.random.choice(Phi_Q.shape[1], k, replace=False)
            Q_hat, w, rmse, _ = la2.lsq(Phi_Q[:,choice], Q_true)

            J_hat = la.multi_dot([mu.T, PI, Q_hat])
            deltaJ.append(abs(J_hat - J_true))
            rmsem += rmse

        err_rand.append(np.mean(deltaJ))
        sdev_err_rand.append(np.std(deltaJ))
        print('random k = %s deltaJ = %s J_hat = %s rmse = %s' % (
                k, np.mean(deltaJ), J_hat, rmsem/trials))

    fig, ax = plt.subplots()
    ax.plot(ks, err_PVF_hat, marker='o', label='gradient PVF')
    ax.errorbar(ks, err_rand, marker='d', yerr=sdev_err_rand, label='random')
    ax.legend(loc='upper right')
    ax.set_xlabel('number of features')
    ax.set_ylabel('delta J')
    plt.savefig('img/Q-function approx gradient.png')
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

def compute(mdp, n_episodes, prox_policy, opt_policy, random_policy):
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

    #---Q-ESTIMATION-PVFs---------------------------------------------------------------------------------------------------

    print('-' * 100)
    print('Q-function estimation with PVF using optimal policy trajectories')
    (eigval_on, PVF_on), (eigval_off, PVF_off) = PVF_estimation(estimator, Q_true, mu, J_true, PI)

    # ---Q-ESTIMATION-PVFs-&-GRADIENT---------------------------------------------------------------------------------------

    print('-' * 100)
    print('Q-function estimation with policy gradient')
    Phi_Q_on, PVF_hat_on = Q_estimation(C, d_sa_mu_hat, Q_true, mu, PI, J_true, PVF_on, eigval_on, mdp_wrap, estimator)
    Phi_Q_off, PVF_hat_off = Q_estimation(C, d_sa_mu_hat, Q_true, mu, PI, J_true, PVF_off, eigval_off, mdp_wrap, estimator)

    #---R-ESTIMATION--------------------------------------------------------------------------------------------------------
    #- -Method 1 - P estimation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print('-' * 100)
    print('\nR estimation - Method 1 - P estimation')
    R_estimation(estimator, mdp_wrap, PI, Phi_Q_on, d_sa_mu, J_true, R, PVF_hat_on)
    R_estimation(estimator, mdp_wrap, PI, Phi_Q_off, d_sa_mu, J_true, R, PVF_hat_off)
    '''
    #- -Method 2 - d_sasa estimation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print('\nR-ESTIMATION - Method 2 - d_sasa estimation')
    d_sasa_hat = np.dot(estimator.d_sas2, PI) + np.eye(PI.shape[1])
    d_sasa_mu_hat = la.multi_dot([np.diag(d_sa_mu_hat), d_sasa_hat])
    Phi_R_2 = la2.nullspace(np.dot(C.T,  d_sasa_mu_hat))
    print('Number of R-features (rank of Phi_R_2) %s' % la.matrix_rank(Phi_R_2))

    R_hat_2, w, rmse = LSestimate(Phi_R_2, R)
    J_hat = np.dot(d_sa_mu,R_hat_2)
    print('Results of LS deltaJ = %s J_hat = %s rmse = %s'% (abs(J_true - J_hat), J_hat, rmse))

    f, axarr = plt.subplots(mdp_wrap.nA, sharex=True)
    for i in range(mdp_wrap.nA):
        axarr[i].plot(np.arange(mdp_wrap.nS), R[mdp_wrap.nA * np.arange(mdp_wrap.nS) + i] - R_hat_1[mdp_wrap.nA * np.arange(mdp_wrap.nS) + i], 'r')
        axarr[i].plot(np.arange(mdp_wrap.nS), R[mdp_wrap.nA * np.arange(mdp_wrap.nS) + i] - R_hat_2[mdp_wrap.nA * np.arange(mdp_wrap.nS) + i], 'b')

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
