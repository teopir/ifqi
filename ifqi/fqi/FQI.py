from __future__ import print_function
import numpy as np
import sklearn.preprocessing as preprocessing
from numpy.matlib import repmat

from ifqi.preprocessors.features import select_features
from ifqi.models.actionregressor import ActionRegressor

"""
This class implements the functions to run Fitted Q-Iteration algorithm.
"""


class FQI:
    def __init__(self, estimator, state_dim, action_dim,
                 discrete_actions, gamma, horizon, features=None, verbose=False):
        """
        Constructor.
        Args:
            estimator (object): the model to be trained
            state_dim (int): state dimensionality
            action_dim (int): action dimensionality
            discrete_actions (list, array): list of discrete actions
            gamma (float): discount factor
            horizon (int): horizon
            scaled (bool, False): true if the input/output are normalized
            features (object, None): kind of features to use
            verbose (int, False): verbosity level

        """
        self._estimator = estimator
        self.gamma = gamma
        self.horizon = horizon

        self.state_dim = state_dim
        self.action_dim = action_dim

        if isinstance(discrete_actions, np.ndarray):
            if len(discrete_actions.shape) > 1:
                assert discrete_actions.shape[1] == action_dim
                assert discrete_actions.shape[0] > 1, \
                    'Error: at least two actions are required'
                self._actions = discrete_actions
            else:
                assert action_dim == 1
                self._actions = np.array(discrete_actions, dtype='float32').T
        elif isinstance(discrete_actions, list):
            assert len(discrete_actions) > 1, \
                'Error: at least two actions are required'
            self._actions = np.array(
                discrete_actions, dtype='float32').reshape(-1, action_dim)
        else:
            raise ValueError(
                'Supported types for discrete_actions are {np.darray, list')

        self.__name__ = "FittedQIteration"
        self._iteration = 0
        self._features = select_features(features)
        self._verbose = verbose

    def _check_states(self, X):
        """
        Check the correctness of the matrix containing the dataset.
        Args:
            X (numpy.array): the dataset
        Returns:
            The matrix containing the dataset reshaped in the proper way.
        """
        return X.reshape(-1, self.state_dim)

    def _preprocess_data(self, sast, r):
        """
        Preprocessing of the dataset. Data are normalized and features are
        computed.
        If inputs are None, no operation is performed and the status of the
        elements associated to
        the dataset are not altered. This means that the instances of sast
        and r stored in the internal state of the class are preserved.
        Args:
            sast (numpy.array): the input in the dataset (state, action,
                                next_state, terminal_flag).
                                Dimensions are (nsamples x nfeatures)
            r (numpy.array): the output in the dataset. Dimensions
                             are (nsamples x 1)
        """
        if sast is not None:
            # get number of samples
            n_samples = sast.shape[0]
            nextstate_idx = self.state_dim + self.action_dim

            sa = sast[:, :nextstate_idx]
            snext = sast[:, nextstate_idx:-1]
            absorbing = sast[:, -1]

            if self._features is not None:
                sa = self._features(sa)

            self._sa = sa
            # Scaling and feature of next states are computed in maxQA
            self._snext = snext

            if isinstance(self._estimator, ActionRegressor):
                self._estimator._actions = np.unique(self._sa[:, -1])

        if r is not None:
            self._r = r.ravel()

        self._absorbing = absorbing

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
        # preprocess new data
        if sast is not None or r is not None:
            self._preprocess_data(sast, r)

        if self._iteration == 0:
            if self._verbose > 0:
                print('Iteration {}'.format(self._iteration + 1))

            y = self._r
        else:
            maxq, maxa = self.maxQA(self._snext, self._absorbing)

            if self._verbose > 0:
                print('Iteration {}'.format(self._iteration + 1))

            if hasattr(self._estimator, 'has_ensembles') \
               and self._estimator.has_ensembles():
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
            print("Starting complete FQI ...")

        # reset iteration count
        self.reset()

        # main loop
        self.partial_fit(sast, r, **kwargs)
        for t in range(1, self.horizon):
            self.partial_fit(sast=None, r=None, **kwargs)

    def maxQA(self, states, absorbing, evaluation=False):
        """
        Computes the maximum Q-function and the associated action
        in the provided states.
        Args:
            states (numpy.array): states to be evaluated.
                                  Dimenions: (nsamples x state_dim)
            absorbing (bool): true if the current state is absorbing.
                              Dimensions: (nsamples x 1)
        Returns:
            Q: the maximum Q-value in each state
            A: the action associated to the max Q-value in each state
        """
        new_state = self._check_states(states)
        n_states = new_state.shape[0]
        n_actions = self._actions.shape[0]

        Q = np.zeros((n_states, n_actions))
        for idx in range(n_actions):
            actions = np.matlib.repmat(self._actions[idx], n_states, 1)

            samples = np.concatenate((new_state, actions), axis=1)

            if self._features is not None:
                samples = self._features.test_features(samples)

            # predict Q-function
            if not evaluation and hasattr(self._estimator, 'has_ensembles') \
               and self._estimator.has_ensembles():
                opt_pars = {'n_actions': n_actions, 'idx': idx}
            else:
                opt_pars = dict()
            predictions = self._estimator.predict(samples, **opt_pars)

            Q[:, idx] = predictions * (1 - absorbing)

        # compute the maximal action
        amax = np.argmax(Q, axis=1)

        # store Q-value and action for each state
        rQ, rA = np.zeros(n_states), np.zeros(n_states)
        for idx in range(n_states):
            rQ[idx] = Q[idx, amax[idx]]
            rA[idx] = self._actions[amax[idx]]

        return rQ, rA

    def draw_action(self, states, absorbing, evaluation=False):
        """
        Compute the action with the highest Q value.
        Args:
            states (numpy.array): the states to be evaluated.
                                  Dimensions: (nsamples x state_dim)
            absorbing (bool): true if the current state is absorbing.
                              Dimensions: (nsamples x 1)
        Returns:
            the argmax and the max Q value
        """
        if self._iteration == 0:
            raise ValueError(
                'The model must be trained before being evaluated')

        _, maxa = self.maxQA(states, absorbing, evaluation)

        return maxa

    def reset(self):
        """
        Reset FQI.
        """
        self._iteration = 0
        self._sa = None
        self._r = None
        self._absorbing = None
        # TODO: reset something else?
