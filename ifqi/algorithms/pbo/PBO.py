import numpy as np
from pybrain.optimization import ExactNES

from ifqi.algorithms.algorithm import Algorithm


class PBO(Algorithm):
    def __init__(self, estimator, state_dim, action_dim,
                 discrete_actions, gamma, horizon,
                 scaled=False, features=None, verbose=False):
        self.optimizer = ExactNES(self.fitness, np.zeros(2), minimize=True,
                                  desiredEvaluation=1e-8)
        super(PBO, self).__init__(estimator, state_dim, action_dim,
                                  discrete_actions, gamma, horizon, scaled,
                                  features, verbose)

    def fit(self, sast=None, r=None):
        if sast is not None or r is not None:
            self._preprocess_data(sast, r)

        old_theta = self._estimator.theta

        rho, score = self.optimizer.learn()
        self._estimator.theta = self.f(rho)

        return (self._estimator.theta,
                np.sum(self._estimator.theta - old_theta) ** 2)

    def fitness(self, rho):
        n_samples = self._sa.shape[0]

        opt_pars = {'f_rho': self.f(rho)}
        Q = self._estimator.predict(self._sa, **opt_pars)
        maxQ, _ = self.maxQA(self._snext, self._absorbing)
        result = np.sum(Q - self._r - self.gamma * maxQ) ** 2
        result /= n_samples

        print(result)

        return result

    def f(self, rho):
        return rho * self._estimator.theta
