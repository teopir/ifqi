import numpy as np
import numpy.linalg as la
from reward_space.utils.sample_estimator import SampleEstimator

class DiscreteEnvSampleEstimator(SampleEstimator):

    def __init__(self, dataset, gamma, state_space, action_space, horizon):

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
        :param horizon: int, maximum length of episode
        '''
        self.dataset = dataset
        self.gamma = gamma
        self.state_space = state_space
        self.action_space = action_space
        self.horizon = horizon
        self._estimate()

    def _estimate(self):
        states = self.dataset[:, 0]
        actions = self.dataset[:, 1]
        rewards = self.dataset[:, 2]
        next_states = self.dataset[:, 3]
        discounts = self.dataset[:, 4]

        nS = len(self.state_space)
        nA = len(self.action_space)

        n_episodes = 0

        P = np.zeros((nS * nA, nS))
        mu = np.zeros(nS)
        d_s_mu = np.zeros(nS)
        d_sa_mu = np.zeros(nS * nA)

        i = 0
        while i < self.dataset.shape[0]:
            s_i = np.argwhere(self.state_space == states[i])
            a_i = np.argwhere(self.action_space == actions[i])
            s_next_i = np.argwhere(self.state_space == next_states[i])

            P[s_i * nA + a_i, s_next_i] += 1
            d_s_mu[s_i] += discounts[i]
            d_sa_mu[s_i * nA + a_i] += discounts[i]

            if i == 0 or self.dataset[i - 1, -1] == 1:
                mu[s_i] += 1
                n_episodes += 1

            i += 1

        d_s_mu[-1] = n_episodes * (1 - self.gamma ** self.horizon) / (1 - self.gamma)  - d_s_mu.sum()
        d_sa_mu[-nA:] = d_s_mu[-1] / nA

        sa_count = P.sum(axis=1)
        sa_count[-nA:] = (n_episodes * self.horizon - self.dataset.shape[0]) / nA
        self.sa_count = sa_count
        P = np.apply_along_axis(lambda x: x / (sa_count + self.tol), axis=0, arr=P)
        mu /= mu.sum()

        d_s_mu /= n_episodes
        d_sa_mu /= n_episodes

        self.P = P
        self.mu = mu
        self.d_s_mu = d_s_mu
        self.d_sa_mu = d_sa_mu
        self.count_sa = sa_count

        self.J = 1.0 / n_episodes * np.sum(rewards * discounts)
