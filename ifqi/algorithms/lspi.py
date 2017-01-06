from __future__ import print_function

import numpy as np

from ifqi.algorithms.algorithm import Algorithm


class LSTDQ(Algorithm):
    def __init__(self, estimator, state_dim, action_dim,
                 discrete_actions, gamma, verbose=False):
        super(LSTDQ, self).__init__(estimator, state_dim, action_dim,
                                    discrete_actions, gamma, None,
                                    verbose)

    def fit(self, sast=None, r=None, **kwargs):
        """
        Run LSTDQ using input data sast and r.

        Args:
            sast (numpy.array): the input in the dataset
            r (numpy.array): the output in the dataset
            **kwargs: additional parameters to be provided to the fit function
                      of the estimator
        """
        if sast is not None:
            next_states_idx = self.state_dim + self.action_dim
            self._sa = sast[:, :next_states_idx]
            self._snext = sast[:, next_states_idx:-1]
            self._absorbing = sast[:, -1]

            self._phi_hat = self._estimator.features.fit_transform(self._sa).T
            # to initialize the regressor
            self._estimator.fit(self._sa, np.zeros((self._sa.shape[0], 1)))

            self._iteration = 1
        if r is not None:
            self._r = r

        best_actions = self.draw_action(self._snext,
                                        self._absorbing).reshape(-1, 1)
        snext_anext = np.concatenate((self._snext, best_actions), axis=1)
        pi_phi_hat = self._estimator.features.transform(snext_anext).T

        A = np.dot(self._phi_hat, (self._phi_hat - self.gamma * pi_phi_hat).T)
        b = np.dot(self._phi_hat, self._r.reshape(-1, 1))

        if np.linalg.matrix_rank(A) == self._phi_hat.shape[0]:
            w = np.linalg.solve(A, b)
        else:
            w = np.dot(np.linalg.pinv(A), b)

        self._estimator.set_weights(w.T)


class LSPI(object):
    """
    This class implements the Least-Squares Policy Iteration (LSPI) algorithm.
    This algorithm is an off-policy batch algorithm that exploits
    the action-values approximation done by the LSTDQ algorithm
    to form an approximate policy-iteration algorithm.

    References
    =========
    [Lagoudakis, Parr. Least-Squares Policy Iteration](http://jmlr.csail.mit.edu/papers/volume4/lagoudakis03a/lagoudakis03a.ps)
    """

    def __init__(self, estimator, state_dim, action_dim,
                 discrete_actions, gamma, epsilon=1e-6, verbose=False):
        self.__name__ = 'LSPI'
        self._epsilon = epsilon
        self._lstdq = LSTDQ(estimator, state_dim, action_dim,
                            discrete_actions, gamma, verbose)

    def fit(self, sast, r, **kwargs):
        """
        Run LSPI using input data sast and r.

        Args:
            sast (numpy.array): the input in the dataset
            r (numpy.array): the output in the dataset
            **kwargs: additional parameters to be provided to the fit function
                      of the estimator
        """
        self._lstdq.fit(sast, r, **kwargs)

        iteration = 1
        delta = np.inf
        while delta >= self._epsilon:
            print('Iteration: %d' % iteration)

            old_w = np.copy(self._lstdq._estimator.get_weights())
            self._lstdq.fit()

            delta = np.linalg.norm(self._lstdq._estimator.get_weights() - old_w)
            print('delta: %f' % delta)
            iteration += 1

    def draw_action(self, states, absorbing, evaluation=False):
        return self._lstdq.draw_action(states, absorbing, evaluation)

    def maxQA(self, states, absorbing, evaluation=False):
        return self._lstdq.maxQA(states, absorbing, evaluation)
