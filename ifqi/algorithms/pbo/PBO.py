from __future__ import print_function

import numpy as np
from builtins import super
from pybrain.optimization import ExactNES

from ifqi.algorithms.algorithm import Algorithm
from keras.models import Sequential
from keras.layers import Dense


class PBO(Algorithm):
    def __init__(self, estimator, state_dim, action_dim,
                 discrete_actions, gamma, horizon,
                 features=None, verbose=False):
        self._regressor_rho = Sequential()
        self._regressor_rho.add(Dense(30, input_shape=(2,), activation='relu'))
        self._regressor_rho.add(Dense(2, activation='linear'))
        self._regressor_rho.compile(optimizer='rmsprop', loss='mse')

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

        self._optimizer = ExactNES(self._fitness, self._get_rho(),
                                   minimize=True, batchSize=100,
                                   maxLearningSteps=0)
        rho, score = self._optimizer.learn()
        self._estimator._regressor.theta = self._f(rho)

        self._iteration += 1

        return (self._estimator._regressor.theta,
                np.sum(self._estimator._regressor.theta - old_theta) ** 2)

    def _fitness(self, rho):
        Q = self._estimator.predict(self._sa, f_rho=self._f(rho))
        maxQ, _ = self.maxQA(self._snext, self._absorbing)

        return np.mean((Q - self._r - self.gamma * maxQ) ** 2)

    def _f(self, rho):
        self._set_rho(rho)
        output = self._regressor_rho.predict(
            np.array([self._estimator._regressor.theta]),
            batch_size=1).ravel()

        return output

    def _get_rho(self):
        rho = self._regressor_rho.get_weights()
        r = list()
        for i in rho:
            r += i.ravel().tolist()

        return np.array(r)

    def _set_rho(self, rho):
        weights = list()
        rho = rho.tolist()
        for l in self._regressor_rho.layers:
            w = l.get_weights()[0]
            b = l.get_weights()[1]
            weights.append(np.array(rho[:w.size]).reshape(w.shape))
            del rho[:w.size]
            weights.append(np.array(rho[:b.size]))
            del rho[:b.size]

        self._regressor_rho.set_weights(weights)
