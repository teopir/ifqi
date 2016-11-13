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
    """
    Constructor.
    Args:
        estimator (object): the model to be trained
        state_dim (int): state dimensionality
        action_dim (int): action dimensionality
        discrete_actions (list, array): list of discrete actions
        gamma (float): discount factor, default=0.9
        horizon (int): horizon
        scaled (bool, None): true if the input/output are normalized
        features (object, None): kind of features to use
        verbose (int, False): verbosity level

    """

    def __init__(self, estimator, state_dim, action_dim,
                 discrete_actions, gamma, horizon,
                 scaled=False, features=None, verbose=False):
        self.estimator = estimator
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
        self.iteration = 0
        self.scaled = scaled
        self.features = select_features(features)
        self.verbose = verbose

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

            if self.scaled:
                # create scaler and fit it
                self._sa_scaler = preprocessing.StandardScaler()
                sa = self._sa_scaler.fit_transform(sa)

            if self.features is not None:
                sa = self.features(sa)

            self.sa = sa
            # Scaling and feature of next states are computed in maxQA
            self.snext = snext

        if r is not None:
            if self.scaled:
                # create scaler and fit it
                self._r_scaler = preprocessing.StandardScaler()
                r = self._r_scaler.fit_transform(r.reshape((-1, 1)))

            self.r = r.ravel()

        self.absorbing = absorbing

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

        # check if the estimator change the structure at each iteration
        adaptive = False

        if hasattr(self.estimator, 'adapt'):
            adaptive = True

        if self.iteration == 0:
            if self.verbose > 0:
                print('Iteration {}'.format(self.iteration + 1))

            y = self.r
        else:
            maxq, maxa = self.maxQA(self.snext, self.absorbing)

            if self.verbose > 0:
                print('Iteration {}'.format(self.iteration + 1))

            if adaptive:
                # update estimator structure
                self.estimator.adapt(iteration=self.iteration)

            y = self.r + self.gamma * maxq

        self.estimator.fit(self.sa, y.ravel(), **kwargs)

        self.iteration += 1

        return self.sa, y

    def fit(self, sast, r, **kwargs):
        """
        Perform steps of FQI using input data sast and r.

        Args:
            sast (numpy.array, None): the input in the dataset
            r (numpy.array, None): the output in the dataset
            **kwargs: additional parameters to be provided to the fit function
                      of the estimator

        Returns:
            sa, y: the preprocessed input and output
        """
        if self.verbose > 0:
            print("Starting complete FQI ...")

        # reset iteration count
        self.reset()

        # main loop
        self.partial_fit(sast, r, **kwargs)
        for t in range(1, self.horizon):
            self.partial_fit(sast=None, r=None, **kwargs)

    def maxQA(self, states, absorbing):
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

        Q = np.zeros((n_states, self._actions.shape[0]))
        for idx in range(self._actions.shape[0]):
            actions = np.matlib.repmat(self._actions[idx], n_states, 1)

            # concatenate [new_state, action] and scalarize them
            if self.scaled:
                samples = self._sa_scaler.transform(np.concatenate((new_state,
                                                                    actions),
                                                                   axis=1))
            else:
                samples = np.concatenate((new_state, actions), axis=1)

            if self.features is not None:
                samples = self.features.test_features(samples)

            # predict Q-function
            predictions = self.estimator.predict(samples)

            Q[:, idx] = predictions * (1 - absorbing)

        # compute the maximal action
        amax = np.argmax(Q, axis=1)

        # store Q-value and action for each state
        rQ, rA = np.zeros(n_states), np.zeros(n_states)
        for idx in range(n_states):
            rQ[idx] = Q[idx, amax[idx]]
            rA[idx] = self._actions[amax[idx]]

        return rQ, rA

    def draw_action(self, states, absorbing):
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
        if self.iteration == 0:
            raise ValueError(
                'The model must be trained before being evaluated')

        _, maxa = self.maxQA(states, absorbing)

        return maxa

    def reset(self):
        """
        Reset FQI.
        """
        self.iteration = 0
        self.sa = None
        self.absorbing = None
        # TODO: reset something else?
