from __future__ import print_function

import numpy as np
import warnings
from builtins import super
from pybrain.optimization import ExactNES

from ifqi.algorithms.algorithm import Algorithm


# pybrain is giving a lot of deprecation warnings
warnings.filterwarnings('ignore', module='pybrain')


class LoggingOptimizerMixin:
    prevX = prevY = None

    def _notify(self):
        x, y = self.bestEvaluable, self.bestEvaluation
        if np.array_equal(x, self.prevX) and y == self.prevY:
            print('.', end='', flush=True)
        else:
            n = self.numLearningSteps
            print('\n{:6} e({}) = {} '.format(n, x, y), end='')
            self.prevX, self.prevY = x, y

    def _bestFound(self):
        print()
        return super()._bestFound()


class LoggingNES(LoggingOptimizerMixin, ExactNES):
    pass



class PBO(Algorithm):
    def __init__(self, estimator, state_dim, action_dim,
                 discrete_actions, gamma, horizon,
                 scaled=False, features=None, verbose=False):
        self._rho = np.zeros(2)

        super(PBO, self).__init__(estimator, state_dim, action_dim,
                                  discrete_actions, gamma, horizon, scaled,
                                  features, verbose)

    def fit(self, sast=None, r=None):
        if sast is not None or r is not None:
            self._preprocess_data(sast, r)

        old_theta = self._estimator.theta

        opt_class = LoggingNES if self._verbose else ExactNES
        self._optimizer = opt_class(self._fitness, self._rho, minimize=True)

        self._rho, score = self._optimizer.learn()
        self._estimator.theta = self._f(self._rho)

        return (self._estimator.theta,
                np.sum(self._estimator.theta - old_theta) ** 2)

    def _fitness(self, rho):
        Q = self._estimator.predict(self._sa, f_rho=self._f(rho))
        maxQ, _ = self.maxQA(self._snext, self._absorbing)

        return np.mean((Q - self._r - self.gamma * maxQ) ** 2)

    def _f(self, rho):
        return rho * self._estimator.theta
