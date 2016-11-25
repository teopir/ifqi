from __future__ import print_function

import numpy as np
import warnings
from builtins import super
from pybrain.optimization import ExactNES

from ifqi.algorithms.algorithm import Algorithm

"""
# pybrain is giving a lot of deprecation warnings
warnings.filterwarnings('ignore', module='pybrain')

"""


class PBO(Algorithm):
    def __init__(self, estimator, state_dim, action_dim,
                 discrete_actions, gamma, horizon,
                 features=None, verbose=False):
        self._rho = np.zeros(2)

        super(PBO, self).__init__(estimator, state_dim, action_dim,
                                  discrete_actions, gamma, horizon,
                                  features, verbose)

    def fit(self, sast=None, r=None):
        if sast is not None:
            next_states_idx = self.state_dim + self.action_dim
            self._sa = sast[:, :next_states_idx]
            self._snext = sast[:, next_states_idx:-1]
            self._absorbing = sast[:, -1]
        if r is not None:
            self._r = r

        old_theta = self._estimator._regressor.theta

        self._optimizer = ExactNES(self._fitness, self._rho, minimize=True)

        self._rho, score = self._optimizer.learn()
        self._estimator._regressor.theta = self._f(self._rho)

        self._iteration += 1

        return (self._estimator._regressor.theta,
                np.sum(self._estimator._regressor.theta - old_theta) ** 2)

    def _fitness(self, rho):
        Q = self._estimator.predict(self._sa, f_rho=self._f(rho))
        maxQ, _ = self.maxQA(self._snext, self._absorbing)

        return np.mean((Q - self._r - self.gamma * maxQ) ** 2)

    def _f(self, rho):
        return rho * self._estimator._regressor.theta
