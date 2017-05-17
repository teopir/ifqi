from __future__ import print_function
from builtins import super

import numpy as np
from pybrain.optimization import ExactNES

from ifqi.algorithms.algorithm import Algorithm


def norm(x, p=2):
    "Norm function accepting both ndarray or tensor as input"
    if p == np.inf:
        return (x ** 2).max()
    x = x if p % 2 == 0 else abs(x)
    return (x ** p).sum() ** (1. / p)


class PBO(Algorithm):
    """
    This class implements the function to run the experimental PBO method.

    """

    def __init__(self, estimator, estimator_rho, state_dim, action_dim,
                 discrete_actions, gamma, learning_steps,
                 batch_size, learning_rate,
                 steps_ahead=1,
                 update_every=1,
                 update_steps=None,
                 norm_value=2,
                 incremental=True, verbose=False):
        """
        Constructor.
        Args:
            estimator_rho (object): the regressor for the function used to
                                    update the q regressor parameters
            learning_steps (int): the number of updates of the q regressor
                                  weights
            batch_size (int): the number of individuals to test in each
                              NES step
            learning rate (float): the value of the learning rate of NES

        """
        self._regressor_rho = estimator_rho
        self._learning_steps = learning_steps
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._incremental = incremental
        self._q_weights_list = list()
        self._norm_value = norm_value
        self._K = steps_ahead
        self._update_every = update_every
        self.steps_per_theta_update = steps_ahead \
            if update_steps is None else max(1, update_steps)
        self.__name__ = 'PBO'
        super(PBO, self).__init__(estimator, state_dim, action_dim,
                                  discrete_actions, gamma, None,
                                  verbose)
        self._rho_values = []

    def fit(self, sast, r):
        """
        Perform a run of PBO using input data sast and r.
        Note that if the dataset does not change between iterations, you can
        provide None inputs after the first iteration.

        Args:
            sast (numpy.array, None): the input in the dataset
            r (numpy.array, None): the output in the dataset
            **kwargs: additional parameters to be provided to the fit function
            of the estimator

        Returns:
            the history of the parameters used to update the q regressor
        """
        self.iteration_best_rho_value = np.inf

        next_states_idx = self.state_dim + self.action_dim
        self._sa = sast[:, :next_states_idx]
        self._snext = sast[:, next_states_idx:-1]
        self._absorbing = sast[:, -1]
        self._r = r

        optimizer = ExactNES(self._fitness, self._get_rho(),
                             minimize=True, batchSize=self._batch_size,
                             learningRate=self._learning_rate,
                             maxLearningSteps=self._learning_steps - 1,
                             importanceMixing=False,
                             maxEvaluations=None)
        optimizer.listener = self.my_listener
        optimizer.learn()
        self._q_weights_list.append(self._get_q_weights())

        return self._q_weights_list

    def my_listener(self, bestEvaluable, bestEvaluation):
        """
        Customized NES listener. It is used to update the parameters of the
        q regressor with the last best one found.
        """
        self._iteration += 1
        if self._verbose:
            print('Iteration: %d' % self._iteration)

        theta = self._get_q_weights()
        self._q_weights_list.append(theta)

        if self._update_every > 0 and self._iteration % self._update_every == 0:
            for _ in range(self.steps_per_theta_update):
                tnext = self._f(self.iteration_best_rho)
                theta = (theta + tnext) if self._incremental else tnext
                self._set_q_weights(theta)

        self._rho_values.append(self.iteration_best_rho)
        if self._verbose:
            print('Global best: %f | Local best: %f' % (
                bestEvaluation, self.iteration_best_rho_value))
        self.iteration_best_rho_value = np.inf

    def _single_step(self, theta0, rho):
        tnext = self._f(rho)
        theta1 = theta0 + tnext if self._incremental else tnext
        self._set_q_weights(theta1)
        q = self._estimator.predict(self._sa)
        self._set_q_weights(theta0)

        max_q, _ = self.maxQA(self._snext, self._absorbing)

        value = norm(q - self._r - self.gamma * max_q, self._norm_value)
        return value, theta1

    def _fitness(self, rho):
        """
        The fitness function used to evaluate the quality of the provided
        individual.

        Args:
            rho (np.array): the individual to test

        Returns:
            the value of the fitness function measured as the mse between
            the Q function computed using the provided individual and the best
            one found at the previous step
        """
        initial_theta = self._get_q_weights()
        theta = self._get_q_weights()
        value = 0.
        for _ in range(self._K):
            tmp, theta = self._single_step(theta, rho)
            value += tmp
        self._set_q_weights(initial_theta)

        if value < self.iteration_best_rho_value:
            self.iteration_best_rho_value = value
            self.iteration_best_rho = rho

        return value

    def _f(self, rho):
        """
        The function computing new q regressor parameters using provided
        individual.

        Args:
            rho (np.array): the individual to consider to computer the new
                            q regressor parameters

        Returns:
            the new q regressor parameters
        """
        self._set_rho(rho)
        output = self._regressor_rho.predict(
            self._get_q_weights().reshape(1, -1)).ravel()

        return output

    def _f2(self, rho, theta):
        """
        The function computing new q regressor parameters using provided
        individual.

        Args:
            rho (np.array): the individual to consider to computer the new
                            q regressor parameters

        Returns:
            the new q regressor parameters
        """
        self._set_rho(rho)
        output = self._regressor_rho.predict(theta.reshape(1, -1)).ravel()

        return output

    def _flat_weights(self, w):
        """
        Flatten the list of weights. It is used, in
        particular, when dealing with the Keras MLP list of weights.

        Args:
            w (list): list of weights

        Returns:
            flattened list of weights
        """
        r = list()
        for i in w:
            r += i.ravel().tolist()

        return np.array(r)

    def _list_weights(self, w, layers):
        """
        Make a list of weights from the provided weights array. It is used, in
        particular, when dealing with the Keras MLP list of weights.

        Args:
            w (np.array): the array of weights to put in a list
            layers (list): the number of layers of the MLP. If the regressor is
                          not a MLP, the function does not use this value
                          and returns a one-dimensional list of weights

        Returns:
            the list of weights
        """
        if layers is not None:
            w = w.tolist()
            weights = list()
            for l in layers:
                W = l.get_weights()[0]
                b = l.get_weights()[1]
                weights.append(np.array(w[:W.size]).reshape(W.shape))
                del w[:W.size]
                weights.append(np.array(w[:b.size]))
                del w[:b.size]

            return weights
        return w

    def _get_rho(self):
        """
        Returns:
             the flattened array of regressor_rho parameters
        """
        return self._flat_weights(self._regressor_rho.get_weights())

    def _set_rho(self, rho):
        """
        Args:
             rho (np.array): the array of parameters to be set in regressor_rho
        """
        layers = self._regressor_rho.layers
        self._regressor_rho.set_weights(self._list_weights(rho, layers))

    def _get_q_weights(self):
        """
        Returns:
             the flattened array of the q regressor weights
        """
        return self._flat_weights(self._estimator._regressor.get_weights())

    def _set_q_weights(self, w):
        """
        Args:
             w (np.array): the array of weights to be set in q regressor
        """
        layers = self._estimator.layers
        self._estimator._regressor.set_weights(self._list_weights(w, layers))
