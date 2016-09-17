from __future__ import print_function
import numpy as np
from numpy.matlib import repmat
import sklearn.preprocessing as preprocessing
from ifqi.preprocessors.features import selectFeatures

"""
This class implements the function to run Fitted Q-Iteration algorithm.

"""


class FQI:
    """
    Constructor.
    Args:
        estimator (object): the model to be trained
        state_dim (int): state dimensionality
        action_dim (int): action dimensionality
        discrete_actions (int): number of actions
        gamma (float): discount factor
        horizon (int): horizon
        scaled (bool): true if the input/output are normalized
        verbose (int): verbosity level
    
    """

    def __init__(self, estimator, state_dim, action_dim,
                 discrete_actions,
                 gamma=0.9, horizon=10,
                 scaled=False, features=None, verbose=0):
        self.estimator = estimator
        self.gamma = gamma
        self.horizon = horizon

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.verbose = verbose
        if isinstance(discrete_actions, np.ndarray):
            assert discrete_actions.shape[1] == action_dim
            assert discrete_actions.shape[0] > 1
            self._actions = discrete_actions
        elif isinstance(discrete_actions, list):
            assert len(discrete_actions) > 1, 'Error: at least two actions are required'
            self._actions = np.array(discrete_actions, dtype='float32').reshape(-1, action_dim)
        else:
            raise ValueError('Supported types for discrete_actions are {np.darray, list, int}')

        self.__name__ = "FittedQIteration"
        self.iteration = 0
        self.scaled = scaled
        self.features = selectFeatures(features) if features is not None else features

    def _check_states(self, X, copy=True):
        """
        Check the correctness of the matrix containing the dataset.
        Args:
            X (numpy.array): the dataset
            copy (bool): ...
        Returns:
            The matrix containing the dataset reshaped in the proper way.
        
        """
        return X.reshape(-1, self.state_dim)

    def _preprocess_data(self, sast=None, r=None):
        """
        Function to normalize data.
        Args:
            sast (numpy.array): the input in the dataset
            r (numpy.array): the output in the dataset

        """

        if sast is not None and r is not None:
            # get number of samples
            nSamples = sast.shape[0]
            action_pos = self.state_dim
            nextstate_pos = self.state_dim + self.action_dim
            absorbing_pos = nextstate_pos + self.state_dim

            sa = np.copy(sast[:, 0:nextstate_pos]).reshape(nSamples, -1)
            snext = sast[:, nextstate_pos:absorbing_pos].reshape(nSamples, -1)
            absorbing = sast[:, absorbing_pos]

            if self.scaled and self.iteration == 0:
                # create scaler and fit it
                self._sa_scaler = preprocessing.StandardScaler()
                sa[:, :-1] = self._sa_scaler.fit_transform(sa[:, :-1])

            if self.features is not None:
                sa = self.features(sa)

            self.sa = sa
            self.snext = snext
            self.absorbing = absorbing
            self.r = r

    def _partial_fit(self, sast=None, r=None, **kwargs):
        """
        Perform a step of FQI.
        Args:
            sast (numpy.array): the input in the dataset
            r (numpy.array): the output in the dataset
            
        Returns:
            the preprocessed input and output
        """
        self._preprocess_data(sast, r)

        # check if the estimator change the structure at each iteration
        adaptive = False
        if hasattr(self.estimator.models[0], 'adapt'):
            adaptive = True

        y = self.r
        if self.iteration == 0:
            if self.verbose > 0:
                print('Iteration {}'.format(self.iteration + 1))

            self.estimator.fit(self.sa, y, **kwargs)
        else:
            maxq, maxa = self.maxQA(self.snext, self.absorbing)
            y = self.r + self.gamma * maxq

            if self.verbose > 0:
                print('Iteration {}'.format(self.iteration + 1))

            if adaptive:
                # update estimator structure
                self.estimator.adapt(iteration=self.iteration)

            self.estimator.fit(self.sa, y, **kwargs)

        self.iteration += 1

        return self.sa, y

    """
    def _fit(self, sast, r, **kwargs):
        '''
        :param sast: array-like, [nbsamples x nfeatures]
         Note that it stores the state, action, nextstate and absorbing flag, this means that
         nfeatures = state_dim*2 + action_dim + 1
        :param r: array-like, [nbsamples x 1]
         Note it stores the reward function associated to the transition sas
        :return: None
        '''
        if self.verbose > 0:
            print("Starting complete FQI ...")

        # reset iteration count
        self.reset()

        # main loop
        self._partial_fit(sast, r)
        for t in range(1, self.horizon):
            self._partial_fit()
    """

    def maxQA(self, states, absorbing=None):
        """
        Computes the maximum Q-function and the associated action
        in the provided states.
        Args:
            states (numpy.array): the current state
            absorbing (bool): true if the current state is absorbing
        Returns:
            the highest Q-value and the associated action
        """
        newstate = self._check_states(states, copy=True)
        nbstates = newstate.shape[0]

        if absorbing is None:
            absorbing = np.zeros((nbstates,))

        Q = np.zeros((nbstates, self._actions.shape[0]))
        for idx in range(self._actions.shape[0]):
            a = self._actions[idx, :].reshape(1, self.action_dim)
            actions = np.matlib.repmat(a, nbstates, 1)

            # concatenate [newstate, action] and scalarize them
            if self.scaled:
                samples = np.concatenate((self._sa_scaler.transform(newstate),
                                          actions),
                                         axis=1)
            else:
                samples = np.concatenate((newstate, actions), axis=1)

            if self.features is not None:
                samples = self.features.testFeatures(samples)

            # predict Q-function
            predictions = self.estimator.predict(samples)

            Q[:, idx] = predictions.ravel()
            Q[:, idx] = Q[:, idx] * (1 - absorbing)

        # compute the maximal action
        amax = np.argmax(Q, axis=1)

        # store Q-value and action for each state
        rQ, rA = np.zeros((nbstates,)), np.zeros((nbstates,))
        for idx in range(nbstates):
            rQ[idx] = Q[idx, amax[idx]]
            rA[idx] = self._actions[amax[idx]]
        # sanity check
        # assert(np.allclose(rQ, Q[np.arange(nbstates), amax]))

        return rQ, rA

    def partial_fit(self, X, y, **kwargs):
        """
        Perform a step of FQI.
        Args:
            X (numpy.array): the input in the dataset
            y (numpy.array): the output in the dataset
            
        Returns:
            the FQI object after one step
            
        """
        self._partial_fit(X, y, **kwargs)
        return self

    """
    def fit(self, X, y, **kwargs):

        self._fit(X, y, **kwargs)
        return self
    """

    def predict(self, states):
        """
        Compute the action with the highest Q value.
        Args:
            states (numpy.array): the current state
        Returns:
            the argmax and the max Q value
            
        """
        if self.iteration == 0:
            raise ValueError('The model must be trained before to be evaluated')

        maxQ, maxa = self.maxQA(states)
        return maxa, maxQ

    def reset(self):
        """
        Reset FQI.
        
        """
        self.iteration = 0
        self.sa = None
        self.snext = None
        self.absorbing = None
        # TODO: reset estimator??
