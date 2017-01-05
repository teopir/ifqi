from __future__ import print_function

import numpy as np

from ifqi.algorithms.algorithm import Algorithm


class LSPI(Algorithm):
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
        super(LSPI, self).__init__(estimator, state_dim, action_dim,
                                   discrete_actions, gamma, None,
                                   verbose)

    def fit(self, sast, r, **kwargs):
        """
        Run LSPI using input data sast and r.

        Args:
            sast (numpy.array): the input in the dataset
            r (numpy.array): the output in the dataset
            **kwargs: additional parameters to be provided to the fit function
                      of the estimator

        Returns:
            w: the computed weights
        """

        # reset iteration count
        self.reset()

        if sast is not None:
            next_states_idx = self.state_dim + self.action_dim
            self._sa = sast[:, :next_states_idx]
            self._snext = sast[:, next_states_idx:-1]
            self._absorbing = sast[:, -1]
        if r is not None:
            self._r = r

        self._iteration = 1

        phi_hat = self._estimator.features(self._sa).T

        self._estimator.fit(self._sa, self._r)
        w = np.zeros((phi_hat.shape[0], 1))
        delta = np.inf
        while delta >= self._epsilon:
            print('Iteration: %d' % self._iteration)

            best_actions = self.draw_action(self._snext,
                                            self._absorbing).reshape(-1, 1)
            snext_anext = np.concatenate((self._snext, best_actions), axis=1)
            pi_phi_hat = self._estimator.features.test_features(snext_anext).T

            A = np.dot(phi_hat, (phi_hat - self.gamma * pi_phi_hat).T)
            b = np.dot(phi_hat, self._r).reshape(-1, 1)

            old_w = np.copy(w)
            if np.linalg.matrix_rank(A) == phi_hat.shape[0]:
                w = np.linalg.solve(A, b)
            else:
                w = np.dot(np.linalg.pinv(A), b)

            self._estimator.set_weights(w.T)

            delta = np.linalg.norm(w - old_w)

            print('delta: %f' % delta)

            self._iteration += 1