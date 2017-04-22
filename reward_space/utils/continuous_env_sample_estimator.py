import numpy as np
from reward_space.utils.sample_estimator import SampleEstimator
import numpy.linalg as la

class ContinuousEnvSampleEstimator(SampleEstimator):

    def __init__(self, dataset, gamma):

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
        '''
        self.dataset = dataset
        self.gamma = gamma
        self.n_samples = dataset.shape[0]
        self._estimate()

    def _estimate(self):
        discounts = self.dataset[:, 4]

        n_episodes = 0

        d_sa_mu = np.zeros(self.n_samples)
        cound_sa = np.ones(self.n_samples) / self.n_samples

        i = 0
        while i < self.n_samples:
            j = i

            d_sa_mu[i] += discounts[i]

            if i == 0 or self.dataset[i - 1, -1] == 1:
                n_episodes += 1

            i += 1

        d_sa_mu /= n_episodes
        self.count_sa = cound_sa
        self.d_sa_mu = d_sa_mu
        self.J = 1.0 / n_episodes * np.sum(self.dataset[:, 2] * self.dataset[:, 4])