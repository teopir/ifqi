from __future__ import print_function

from ifqi.algorithms.algorithm import Algorithm


class FQI(Algorithm):
    """
    This class implements the Fitted Q-Iteration algorithm.
    This algorithm is an off-policy batch algorithm that works with finite
    action spaces.
    It exploits the Bellman operator to build a dataset of Q-values from which
    a regressor is trained.

    References
    ==========
    [Ernst, Geurts, Wehenkel. Tree-Based Batch Mode Reinforcement Learning](http://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf)
    """

    def __init__(self, estimator, state_dim, action_dim,
                 discrete_actions, gamma, horizon, verbose=False):
        self.__name__ = 'FQI'
        super(FQI, self).__init__(estimator, state_dim, action_dim,
                                  discrete_actions, gamma, horizon,
                                  verbose)

    def partial_fit(self, sast=None, r=None, **kwargs):
        """
        Perform a step of FQI using input data sast and r.
        Note that if the dataset does not change between iterations, you can
        provide None inputs after the first iteration.

        Args:
            sast (numpy.array, None): the input in the dataset
            r (numpy.array, None): the output in the dataset
            **kwargs: additional parameters to be provided to the fit function
            of the estimator

        Returns:
            sa, y: the preprocessed input and output
        """
        if sast is not None:
            next_states_idx = self.state_dim + self.action_dim
            self._sa = sast[:, :next_states_idx]
            self._snext = sast[:, next_states_idx:-1]
            self._absorbing = sast[:, -1]
        if r is not None:
            self._r = r

        if self._iteration == 0:
            if self._verbose > 0:
                print('Iteration {}'.format(self._iteration + 1))

            y = self._r
        else:
            if self._verbose > 0:
                print('Iteration {}'.format(self._iteration + 1))

            maxq, maxa = self.maxQA(self._snext, self._absorbing)

            if hasattr(self._estimator, 'adapt'):
                # update estimator structure
                self._estimator.adapt(iteration=self._iteration)

            y = self._r + self.gamma * maxq

        self._estimator.fit(self._sa, y.ravel(), **kwargs)

        self._iteration += 1

        return self._sa, y

    def fit(self, sast, r, **kwargs):
        """
        Perform steps of FQI using input data sast and r.

        Args:
            sast (numpy.array): the input in the dataset
            r (numpy.array): the output in the dataset
            **kwargs: additional parameters to be provided to the fit function
                      of the estimator

        Returns:
            sa, y: the preprocessed input and output
        """
        if self._verbose > 0:
            print("Starting complete run...")

        # reset iteration count
        self.reset()

        # main loop
        self.partial_fit(sast, r, **kwargs)
        for t in range(1, self.horizon):
            self.partial_fit(sast=None, r=None, **kwargs)
