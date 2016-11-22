import numpy as np
from ifqi.algorithms.algorithm import Algorithm


class PBO(Algorithm):
    def __init__(self, estimator, state_dim, action_dim,
                 discrete_actions, gamma, horizon,
                 scaled=False, features=None, verbose=False):
        super(PBO, self).__init__(estimator, state_dim, action_dim,
                                  discrete_actions, gamma, horizon, scaled,
                                  features, verbose)

    def fitness(self, rho):
        n_samples = self._training_set.shape[0]
        result = 0
        for i in range(n_samples):
            m = 0
            s, a, r, ns = self._training_set[i, :4]

            Q = self.Q(ns, self._actions, self.theta)
            Q = Q * (1 - self._training_set[i, -2])  # absorbing states
            result += self.Q(s, a, self.f(rho)) - r - self.gamma * np.max(Q)

        return result ** 2 / n_samples

    def Q(self, s, a, theta):
        k, b = theta
        return b - (a - k * s) ** 2

    def f(self, rho):
        return rho * self.theta
