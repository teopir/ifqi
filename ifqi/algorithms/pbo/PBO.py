from __future__ import print_function
from builtins import super

import numpy as np
from pybrain.optimization import ExactNES

from ifqi.algorithms.algorithm import Algorithm


class PBO(Algorithm):
    """
    This class implements the function to run the experimental PBO method.

    """
    def __init__(self, estimator, estimator_rho, state_dim, action_dim,
                 discrete_actions, gamma, learning_steps,
                 batch_size, learning_rate, verbose=False):
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
        self._q_weights_list = list()
        super(PBO, self).__init__(estimator, state_dim, action_dim,
                                  discrete_actions, gamma, None,
                                  verbose)

    def fit(self, sast=None, r=None):
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
        if sast is not None:
            next_states_idx = self.state_dim + self.action_dim
            self._sa = sast[:, :next_states_idx]
            self._snext = sast[:, next_states_idx:-1]
            self._absorbing = sast[:, -1]
        if r is not None:
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

    def my_listener(self, bestEvaluable, _):
        """
        Customized NES listener. It is used to update the parameters of the
        q regressor with the last best one found.

        Args:
            bestEvaluable (np.array): the array containing the last best
                                      parameters for the q regressor found
        """
        self._iteration += 1
        print('Iteration: %d' % self._iteration)
        self._q_weights_list.append(self._get_q_weights())
        new_q_weights = self._f(bestEvaluable)
        self._set_q_weights(new_q_weights)

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
        old_q_weights = self._get_q_weights()
        self._set_q_weights(self._f(rho))
        q = self._estimator.predict(self._sa)
        self._set_q_weights(old_q_weights)

        max_q, _ = self.maxQA(self._snext, self._absorbing)

        return np.mean((q - self._r - self.gamma * max_q) ** 2)

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
