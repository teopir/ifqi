import numpy as np
import numpy.linalg as la

class DiscreteMdpWrapper(object):
    def __init__(self, mdp, episodic=False):
        self.mdp = mdp
        self.episodic = episodic
        self._compute_mdp_parameters()
        if episodic:
            self._fix_episodic()

    def _compute_mdp_parameters(self):
        nS = self.mdp.observation_space.n
        nA = self.mdp.action_space.n
        self.state_space = range(nS)
        self.action_space = range(nA)
        P = np.zeros(shape=(nS * nA, nS))
        R_sas = np.zeros(shape=(nS * nA, nS))
        D = np.zeros(shape=(nS * nA, nS), dtype=np.bool)
        mu = self.mdp.isd
        for s, v in self.mdp.P.items():
            for a, w in v.items():
                for p, s1, r, d in w:
                    P[s * nA + a, s1] = p
                    R_sas[s * nA + a, s1] = r
                    D[s * nA + a, s1] = d
        R_sa = (R_sas * P).sum(axis=1)
        gamma = self.mdp.gamma
        horizon = self.mdp.horizon

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
        '''
        In order to use the matrix notation for an episodic MDP it is
        necessary to add a fake absorbing state with 0 reward reached
        whenever the terminal states are reached
        '''
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
        # return la.multi_dot([pinve, self.compute_d_ss(), self.PI])
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