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
                 discrete_actions, gamma, verbose=False):
        self.__name__ = 'LSPI'
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

        phi_hat = self._estimator.features(self._sa).T
        best_actions = self.draw_action(self._snext, self._absorbing)
        snext_anext = np.concatenate((self._snext, best_actions), axis=1)
        pi_phi_hat = self._estimator.features(snext_anext)

        A = phi_hat * (phi_hat - self.gamma * pi_phi_hat).T
        b = phi_hat * self._r

        if np.linalg.matrix_rank(A) == phi_hat.shape[0]:
            w = np.linalg.solve(A, b)
        else:
            w = np.linalg.pinv(A) * b

        self._estimator.set_weights(w)