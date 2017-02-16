from __future__ import print_function
from ifqi.envs import TaxiEnv
from policy import  TaxiEnvPolicy, TaxiEnvPolicyStateParameter, TaxiEnvPolicy2StateParameter, TaxiEnvPolicyOneParameter, TaxiEnvPolicy2Parameter
from ifqi.evaluation import evaluation
import numpy as np
import numpy.linalg as la
import scipy.linalg as spla
#from rank_nullspace import nullspace
import matplotlib.pyplot as plt
import statsmodels.api as sm


class DiscereEnvSampleEstimator(object):

    tol = 1e-24 #to avoid divisions by zero

    def __init__(self, dataset, gamma, state_space, action_space):

        '''
        Works only for discrete mdps.
        :param dataset: numpy array (n_samples,7) of the form
            dataset[:,0] = current state
            dataset[:,1] = current action
            dataset[:,2] = reward
            dataset[:,3] = next state
            dataset[:,4] = discount
            dataset[:,5] = a flag indicating whether the reached state is absorbing
            dataset[:,6] = a flag indicating whether the episode is finished (absorbing state
                           is reached or the time horizon is met)
        :param gamma: discount factor
        :param state_space: numpy array
        :param action_space: numpy array
        '''
        self.dataset = dataset
        self.gamma = gamma
        self.state_space = state_space
        self.action_space = action_space
        self._estimate()
    
    def _estimate(self):
        states = self.dataset[:, 0]
        actions = self.dataset[:, 1]
        next_states = self.dataset[:, 3]
        discounts = self.dataset[:, 4]

        nS = len(self.state_space)
        nA = len(self.action_space)

        n_episodes = 0

        P = np.zeros((nS * nA, nS))
        mu = np.zeros(nS)
        
        d_s_mu = np.zeros(nS)
        d_sa_mu = np.zeros(nS * nA)

        d_sas2 = np.zeros((nS * nA, nS))
        d_sasa = np.zeros((nS * nA, nS * nA))
        d_sasa2 = np.zeros((nS * nA, nS * nA))

        d_sasa_mu = np.zeros((nS * nA, nS * nA))

        i = 0
        while i < self.dataset.shape[0]:
            j = i
            s_i = np.argwhere(self.state_space == states[i])
            a_i = np.argwhere(self.action_space == actions[i])
            s_next_i = np.argwhere(self.state_space == next_states[i])
            
            P[s_i * nA + a_i, s_next_i] += 1
            d_s_mu[s_i] += discounts[i]
            d_sa_mu[s_i * nA + a_i] += discounts[i]
            
            if i == 0 or self.dataset[i-1, -1] == 1:
                mu[s_i] += 1
                n_episodes += 1

            while j < self.dataset.shape[0] and self.dataset[j, -1] == 0:
                s_j = np.argwhere(self.state_space == states[j])
                a_j = np.argwhere(self.action_space == actions[j])
                if j > i:
                    d_sas2[s_i * nA + a_i, s_j] += discounts[j] / discounts[i]
                    d_sasa2[s_i * nA + a_i, s_j * nA + a_j] += discounts[j] / discounts[i]
                d_sasa[s_i * nA + a_i, s_j * nA + a_j] += discounts[j] / discounts[i]
                d_sasa_mu[s_i * nA + a_i, s_j * nA + a_j] += discounts[j]
                j += 1

            if j < self.dataset.shape[0]:
                s_j = np.argwhere(self.state_space == states[j])
                a_j = np.argwhere(self.action_space == actions[j])
                if j > i:
                    d_sas2[s_i * nA + a_i, s_j] += discounts[j] / discounts[i]
                    d_sasa2[s_i * nA + a_i, s_j * nA + a_j] += discounts[j] / discounts[i]
                d_sasa[s_i * nA + a_i, s_j * nA + a_j] += discounts[j] / discounts[i]
                d_sasa_mu[s_i * nA + a_i, s_j * nA + a_j] += discounts[j]

            i += 1

        sa_count = P.sum(axis=1)
        self.sa_count = sa_count
        s_count = P.sum(axis=0)
        P = np.apply_along_axis(lambda x: x / (sa_count + self.tol), axis=0, arr=P)
        
        mu /= mu.sum()

        d_s_mu /= n_episodes
        d_sa_mu /= n_episodes
        d_sas2 = d_sas2 / (sa_count[:, np.newaxis] + self.tol)
        d_sasa2 = d_sasa2 / (sa_count[:, np.newaxis] + self.tol) + np.eye(nS * nA)
        d_sasa /= (sa_count[:,np.newaxis] + self.tol)

        d_sasa_mu /= n_episodes

        self.P = P
        self.mu = mu
        self.d_s_mu = d_s_mu
        self.d_sa_mu = d_sa_mu
        self.d_sas2 = d_sas2
        self.d_sasa = d_sasa
        self.d_sasa2 = d_sasa2

        self.d_sasa_mu = d_sasa_mu

        self.J = 1.0 / n_episodes * np.sum(self.dataset[:, 2] * self.dataset[:, 4])


class DiscreteMdpWrapper(object):
    
    def __init__(self, mdp, episodic=False):
        self.mpd = mdp
        self.episodic = episodic
        self._compute_mdp_parameters()
        if episodic:
            self._fix_episodic()

    
    def _compute_mdp_parameters(self):
        nS = mdp.observation_space.n
        nA = mdp.action_space.n
        self.state_space = range(nS)
        self.action_space = range(nA)
        P = np.zeros(shape=(nS * nA, nS))
        R_sas = np.zeros(shape=(nS * nA, nS))
        D = np.zeros(shape=(nS * nA, nS), dtype=np.bool)
        mu = mdp.isd
        for s, v in mdp.P.items():
            for a, w in v.items():
                for p, s1, r, d in w:
                    P[s * nA + a, s1] = p
                    R_sas[s * nA + a, s1] = r
                    D[s * nA + a, s1] = d
        R_sa = (R_sas * P).sum(axis=1)
        gamma = mdp.gamma
        horizon = mdp.horizon
        
        self.nS = nS
        self.nA = nA
        self.P = P
        self.R_sas = R_sas
        self.R = R_sa
        self.D = D
        self.mu = mu
        self.gamma = gamma
        self.horizon = horizon

    def _fix_episodic(self):
        P = np.hstack([self.P, np.zeros((self.P.shape[0], 1))])
        R_sas = np.hstack([self.R_sas, np.zeros((self.R_sas.shape[0], 1))])
        for i, j in zip(*self.D.nonzero()):
            P[i, -1] = P[i, j]
            P[i, j] = 0.
            R_sas[i, -1] = R_sas[i, j]
            R_sas[i, j] = 0.
        R_sa = (R_sas * P).sum(axis=1)
        mu = np.concatenate([self.mu, [0]])

        self.non_ep_P = self.P
        self.non_ep_R_sas = self.R_sas
        self.non_ep_R = self.R
        self.non_ep_mu = self.mu

        self.P = P
        self.R_sas = R_sas
        self.R = R_sa
        self.mu = mu

    def set_policy(self, PI):

        if self.episodic:
            self.non_ep_PI = PI
            self.PI = np.vstack([PI, np.zeros((1, PI.shape[1]))])
        else:
            self.PI = PI

    def compute_d_s_mu(self):
        A = np.eye(self.PI.shape[0]) - self.gamma * np.dot(self.PI, self.P).T
        return la.solve(A, self.mu)

    def compute_d_sa_mu(self):
        return np.dot(self.PI.T, self.compute_d_s_mu())
    
    def compute_d_ss(self):
        A = np.eye(self.PI.shape[0]) - self.gamma * np.dot(self.PI, self.P)
        return la.inv(A)

    def compute_d_sasa(self):
        #return la.multi_dot([pinve, self.compute_d_ss(), self.PI])
        A = np.eye(self.PI.shape[1]) - self.gamma * np.dot(self.P, self.PI)
        return la.inv(A)
    
    def compute_d_ss_mu(self):
        return np.dot(np.diag(self.compute_d_s_mu()), self.compute_d_ss())

    def compute_d_sasa_mu(self):
        return np.dot(np.diag(self.compute_d_sa_mu()), self.compute_d_sasa())

    def compute_Q_function(self):
        A = np.eye(self.P.shape[0]) - self.gamma * np.dot(self.P, self.PI)
        return la.solve(A, self.R)

    def compute_V_function(self):
        A = np.eye(self.P.shape[1]) - self.gamma * np.dot(self.PI, self.P)
        b = np.dot(self.PI, self.R)
        return la.solve(A, b)

    def compute_J(self):
        V = self.compute_V_function()
        return np.dot(V, self.mu)

def nullspace(A, criterion='rank', atol=1e-13, rtol=0):

    '''
    Computes the null space of matrix A
    :param A: the matrix
    :param criterion: 'rank' or 'tol' If 'rank' it uses the rank of matrix A
                      to determine the rank of the null space, otherwise it uses
                      the tolerance
    :param atol:    absolute tolerance
    :param rtol:    relative tolerance
    :return:        the matrix whose columns are the null space of A
    '''

    A = np.atleast_2d(A)
    u, s, vh = la.svd(A)
    if criterion == 'tol':
        tol = max(atol, rtol * s[0])
        nnz = (s >= tol).sum()
    else:
        nnz = la.matrix_rank(A)
    ns = vh[nnz:].conj().T
    return ns

def LSestimate(X, Q_true):
    '''
    Performs LS estimation of the Q function starting from the orthonormal
    basis X and the target Q_true
    '''
    w, residuals, rank, _ =  la.lstsq(X, Q_true)
    rmse = np.sqrt(residuals/X.shape[0])
    Q_hat = X.dot(w)
    return Q_hat, w, rmse


def compute(mdp, n_episodes, prox_policy, opt_policy):
    dataset = evaluation.collect_episodes(mdp, prox_policy, n_episodes)

    mdp_wrap = DiscreteMdpWrapper(mdp,episodic=True)

    PI = prox_policy.get_distribution()
    PI_opt = opt_policy.get_distribution()

    estimator = DiscereEnvSampleEstimator(dataset,
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
    d_sasa_mu = mdp_wrap.compute_d_sasa_mu()
    R = mdp_wrap.non_ep_R
    J_true = mdp_wrap.compute_J()
    Q_true = mdp_wrap.compute_Q_function()

    C = prox_policy.gradient_log_pdf()

    #Sample estimates
    d_s_mu_hat = estimator.d_s_mu
    d_sa_mu_hat = np.dot(PI.T, d_s_mu_hat)
    J_hat = estimator.J

    print('Expected reward opt det policy J = %g' % J_opt)
    print('True expected reward approx opt policy J_true = %g' % J_true)
    print('Estimated expected reward approx opt policy J_true = %g' % J_hat)

    grad_J_hat = 1.0 / n_episodes * la.multi_dot([C.T, np.diag(d_sa_mu_hat), Q_true])
    grad_J_true = la.multi_dot([C.T, np.diag(d_sa_mu), Q_true])
    print('Dimension of the subspace %s' % la.matrix_rank(np.dot(C.T, np.diag(d_sa_mu_hat))))
    print('True policy gradient (2-norm) %s' % la.norm(grad_J_hat))
    print('Estimated policy gradient (2-norm)%s' % la.norm(grad_J_true))

    #---Q-ESTIMATION--------------------------------------------------------------------------------------------------------
    print('\nQ-ESTIMATION')

    Phi_Q = nullspace(np.dot(C.T, np.diag(d_sa_mu_hat)))
    print('Number of Q-features (rank of Phi_Q) %s' % la.matrix_rank(Phi_Q))

    Q_hat, w, rmse = LSestimate(Phi_Q, Q_true)
    J_hat = la.multi_dot([mu.T, PI, Q_hat])
    print('Results of LS deltaJ = %s J_hat = %s rmse = %s'% (abs(J_true - J_hat), J_hat, rmse))

    #---R-ESTIMATION--------------------------------------------------------------------------------------------------------
    #- -Method 1 - P estimation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print('\nR-ESTIMATION - Method 1 - P estimation')
    P_hat = estimator.P
    Phi_R_1 = np.dot(np.eye(mdp_wrap.nA * mdp_wrap.nS) - mdp_wrap.gamma * np.dot(P_hat, PI), Phi_Q)
    print('Number of R-features (rank of Phi_R_1) %s' % la.matrix_rank(Phi_R_1))

    R_hat_1, w, rmse = LSestimate(Phi_R_1, R)
    J_hat = np.dot(d_sa_mu,R_hat_1)
    print('Results of LS deltaJ = %s J_hat = %s rmse = %s'% (abs(J_true - J_hat), J_hat, rmse))

    #- -Method 2 - d_sasa estimation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print('\nR-ESTIMATION - Method 2 - d_sasa estimation')
    d_sasa_hat = np.dot(estimator.d_sas2, PI) + np.eye(PI.shape[1])
    d_sasa_mu_hat = la.multi_dot([np.diag(d_sa_mu_hat), d_sasa_hat])
    Phi_R_2 = nullspace(np.dot(C.T,  d_sasa_mu_hat))
    print('Number of R-features (rank of Phi_R_2) %s' % la.matrix_rank(Phi_R_2))

    R_hat_2, w, rmse = LSestimate(Phi_R_2, R)
    J_hat = np.dot(d_sa_mu,R_hat_2)
    print('Results of LS deltaJ = %s J_hat = %s rmse = %s'% (abs(J_true - J_hat), J_hat, rmse))

    f, axarr = plt.subplots(mdp_wrap.nA, sharex=True)
    for i in range(mdp_wrap.nA):
        axarr[i].plot(np.arange(mdp_wrap.nS), R[mdp_wrap.nA * np.arange(mdp_wrap.nS) + i] - R_hat_1[mdp_wrap.nA * np.arange(mdp_wrap.nS) + i], 'r')
        axarr[i].plot(np.arange(mdp_wrap.nS), R[mdp_wrap.nA * np.arange(mdp_wrap.nS) + i] - R_hat_2[mdp_wrap.nA * np.arange(mdp_wrap.nS) + i], 'b')


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
