from __future__ import print_function
from ifqi.envs import TaxiEnv
from policy import  TaxiEnvPolicy, TaxiEnvPolicyStateParameter, TaxiEnvPolicy2StateParameter, TaxiEnvPolicyOneParameter, TaxiEnvPolicy2Parameter
from ifqi.evaluation import evaluation
import numpy as np
import numpy.linalg as LA
from rank_nullspace import nullspace
import matplotlib.pyplot as plt

def compute_mdp_parameters(mdp, fix_episodic=True):
    nS = mdp.observation_space.n
    nA = mdp.action_space.n
    P = np.zeros(shape=(nS*nA, nS))
    R_sas = np.zeros(shape=(nS*nA, nS))
    D = np.zeros(shape=(nS*nA, nS), dtype=np.bool)
    mu = mdp.isd
    for s,v in mdp.P.items():
        for a,w in v.items():
            for p,s1,r,d in w:
                P[s*nA + a, s1] = p
                R_sas[s*nA + a, s1] = r
                D[s*nA + a, s1] = d
    R_sa = (R_sas * P).sum(axis=1)
    gamma = mdp.gamma
    horizon = mdp.horizon

    return P, R_sas, R_sa, D, mu, gamma, horizon

def fix_episodic(P, R_sas, R_sa, D, mu, PI):
    P = np.hstack([P, np.zeros((P.shape[0],1))])
    R_sas = np.hstack([R_sas, np.zeros((R_sas.shape[0],1))])
    cont = 0
    for i, j in zip(*D.nonzero()):
        P[i, -1] = P[i,j]
        P[i,j] = 0.
        R_sas[i, -1] = R_sas[i,j]
        R_sas[i,j] = 0.
    R_sa = (R_sas * P).sum(axis=1)
    PI = np.vstack([PI, np.zeros((1,PI.shape[1]))])
    mu = np.concatenate([mu, [0]])
    return P, R_sas, R_sa, D, mu, PI


def compute_d_s(P, gamma, PI, mu):
    A = np.eye(P.shape[1]) - gamma * np.dot(PI, P).T
    return LA.solve(A, mu)

def compute_d_sa(P, gamma, PI, mu):
    return np.dot(PI.T, compute_d_s(P, gamma, PI, mu))

def compute_Q_function(P, R, gamma, PI):
    A = np.eye(P.shape[0]) - gamma * np.dot(P, PI)
    return LA.solve(A, R)

def compute_V_function(P, R, gamma, PI):
    A = np.eye(P.shape[1]) - gamma * np.dot(PI, P)
    b = np.dot(PI, R)
    return LA.solve(A, b)

def compute_J(P, R, gamma, PI, mu):
    V = compute_V_function(P, R, gamma, PI)
    return np.dot(V, mu)

    '''
    With policy gradient
    delta_s = compute_delta_s(P, gamma, PI, mu)
    return la.multi_dot([delta_s, PI, R])
    '''

def compute_counts(dataset, s_idx, a_idx, disc_idx, state_space, action_space):
    states = dataset[:,s_idx]
    actions = dataset[:,a_idx]
    discounts = dataset[:,disc_idx]

    nS = len(state_space)
    nA = len(action_space)

    d_s = np.zeros(nS)
    d_a = np.zeros(nA)
    d_sa = np.zeros(nS*nA)

    for s,a,disc in zip(states,actions,discounts):
        s_i = np.argwhere(state_space == s)
        a_i = np.argwhere(action_space == a)
        d_s[s_i] += disc
        d_a[a_i] += disc
        d_sa[s_i*nA + a_i] += disc

    return d_s, d_a, d_sa

def estimate_Q(X, Q_true):
    '''
    Performs LS estimation of the Q function starting from the orthonormal
    basis X and the target Q_true
    '''
    w, residuals, rank, _ =  LA.lstsq(X, Q_true)
    rmse = np.sqrt(residuals/X.shape[0])
    Q_hat = X.dot(w)
    return Q_hat, w, rmse


def compute(mdp, n_episodes, prox_policy, opt_policy):
    dataset = evaluation.collect_episodes(mdp, prox_policy, n_episodes)

    P, R_sas, R, D, mu, gamma, horizon = compute_mdp_parameters(mdp)
    PI = prox_policy.get_distribution()
    PI_opt = opt_policy.get_distribution()

    P2, _, R2, D2, mu2, PI2 = fix_episodic(P, R_sas, R, D, mu, PI)
    _, _, _, _, _, PI2_opt = fix_episodic(P, R_sas, R, D, mu, PI_opt)

    d_s_hat, d_a_hat, d_sa_hat1 = compute_counts(dataset, 0, 1, 4, range(mdp.observation_space.n),
                                                 range(mdp.action_space.n))
    d_sa_hat2 = np.dot(PI.T, d_s_hat)
    d_sa = compute_d_sa(P2, gamma, PI2, mu2)
    Q_true = compute_Q_function(P2, R2, gamma, PI2)
    C = prox_policy.gradient_log_pdf()

    J_opt = compute_J(P2, R2, gamma, PI2_opt, mu2)
    J_true = compute_J(P2, R2, gamma, PI2, mu2)
    J_hat = 1.0 / n_episodes * np.sum(dataset[:, 2] * dataset[:, 4])

    print('Optimal expected reward (deterministic policy) %g' % J_opt)
    print('True expected reward (our policy) %g' % J_true)
    print('Estimated expected reward (our policy) %g' % J_hat)

    grad_J_hat = 1.0 / n_episodes * LA.multi_dot([C.T, np.diag(d_sa_hat2), Q_true])
    grad_J_true = LA.multi_dot([C.T, np.diag(d_sa), Q_true])
    print('Dimension of the subspace %s' % LA.matrix_rank(np.dot(C.T, np.diag(d_sa_hat2))))
    print('True policy gradient %s' % LA.norm(grad_J_hat))
    print('Estimated policy gradient %s' % LA.norm(grad_J_true))

    X = nullspace(np.dot(C.T, np.diag(d_sa_hat2)))
    X_true = nullspace(np.dot(C.T, np.diag(d_sa)))
    print('Rank X %s' % LA.matrix_rank(X))

    print('R-estimation')
    R_hat, w, rmse = estimate_Q(X, R)
    J_hat = np.dot(d_sa,R_hat)
    error = np.abs(R - R_hat)
    mae = np.mean(error)
    error_rel = np.abs((R - R_hat) / (R + 1e-8))
    mare = np.mean(error_rel)
    print('Results of LS deltaJ = %s J_hat = %s rmse = %s mae = %s mare = %s' % (J_true - J_hat, J_hat, rmse, mae, mare))


    print('Q-estimation')
    Q_hat, w, rmse = estimate_Q(X, Q_true)
    J_hat = LA.multi_dot([mu.T, PI, Q_hat])
    error = np.abs(Q_true - Q_hat)
    mae = np.mean(error)
    error_rel = np.abs((Q_true - Q_hat) / (Q_true + 1e-8))
    mare = np.mean(error_rel)
    print('Results of LS deltaJ = %s J_hat = %s rmse = %s mae = %s mare = %s' % (J_true - J_hat, J_hat, rmse, mae, mare))

    # ------------------------------------Plots-------------------------------------------------------------------
    '''
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].stem(np.arange(6), PI_opt[0][:6], '-.')
    axarr[1].stem(np.arange(6), PI[0][:6], '-.')
    plt.show()
    '''

    f, axarr = plt.subplots(6, sharex=True)
    for i in range(6):
        axarr[i].plot(np.arange(500), Q_true[6 * np.arange(500) + i] - Q_hat[6 * np.arange(500) + i])
        #axarr[i].scatter(np.arange(500), Q_true[6 * np.arange(500) + i], c='b', marker='o')
        #axarr[i].scatter(np.arange(500), Q_hat[6 * np.arange(500) + i], c='r', marker='*')


mdp = TaxiEnv()
n_episodes = 1000

opt_policy = TaxiEnvPolicy()

print('1 GLOBAL PARAMETER')
prox_policy = TaxiEnvPolicyOneParameter(3, 0.2, 3)
compute(mdp, n_episodes, prox_policy, opt_policy)

print('\n2 GLOBAL PARAMETERS')
prox_policy = TaxiEnvPolicy2Parameter(sigma=0.02)
compute(mdp, n_episodes, prox_policy, opt_policy)

print('\n1 STATE PARAMETER')
prox_policy = TaxiEnvPolicyStateParameter(3, 0.2, 3)
compute(mdp, n_episodes, prox_policy, opt_policy)

print('\n2 STATE PARAMETERS')
prox_policy = TaxiEnvPolicy2StateParameter(sigma=0.02)
compute(mdp, n_episodes, prox_policy, opt_policy)
plt.show()