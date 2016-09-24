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
        discrete_actions (list, array): list of discrete actions
        gamma (float): discount factor, default=0.9
        horizon (int): horizon, default=10
        scaled (bool): true if the input/output are normalized, default=None
        verbose (int): verbosity level, default=0

    """

    def __init__(self, estimator, stateDim, actionDim,
                 discreteActions,
                 gamma=0.9, horizon=10,
                 scaled=False, features=None, verbose=0):
        self.estimator = estimator
        self.gamma = gamma
        self.horizon = horizon

        self.stateDim = stateDim
        self.actionDim = actionDim

        self.verbose = verbose
        if isinstance(discreteActions, np.ndarray):
            assert discreteActions.shape[1] == actionDim
            assert discreteActions.shape[0] > 1, \
                'Error: at least two actions are required'
            self._actions = discreteActions
        elif isinstance(discreteActions, list):
            assert len(discreteActions) > 1, \
                'Error: at least two actions are required'
            self._actions = np.array(
                discreteActions, dtype='float32').reshape(-1, actionDim)
        else:
            raise ValueError(
                'Supported types for discrete_actions are \
                {np.darray, list, int}')

        self.__name__ = "FittedQIteration"
        self.iteration = 0
        self.scaled = scaled
        self.features = \
            selectFeatures(features) if features is not None else features

    def _checkStates(self, X, copy=True):
        """
        Check the correctness of the matrix containing the dataset.
        Args:
            X (numpy.array): the dataset
            copy (bool): ...
        Returns:
            The matrix containing the dataset reshaped in the proper way.

        """
        return X.reshape(-1, self.stateDim)

    def _checkActions(self, X, copy=True):
        """
        Check the correctness of the matrix containing the dataset.
        Args:
            X (numpy.array): the dataset
            copy (bool): ...
        Returns:
            The matrix containing the dataset reshaped in the proper way.

        """
        return X.reshape(-1, self.actionDim)

    def _preprocessData(self, sast=None, r=None):
        """
        Preprocessing of the dataset. Data are normalized and features are computed.
        If inputs are None, no operation is performed and the status of the elements associated to
        the dataset are not altered. This means that the instances of ``sast`` and ``r`` stored in the internal
        state of the class are preserved.
        Args:
            sast (numpy.array, None): the input in the dataset (state, action, next_state, terminal_flag).
            Dimensions are (nsamples x nfeatures)
            r (numpy.array, None): the output in the dataset. Dimensions are (nsamplex x 1)

        """

        if sast is not None and r is not None:
            # get number of samples
            nSamples = sast.shape[0]
            actionPos = self.stateDim
            nextstatePos = self.stateDim + self.actionDim
            absorbingPos = nextstatePos + self.stateDim

            sa = np.copy(sast[:, 0:nextstatePos]).reshape(nSamples, -1)
            snext = sast[:, nextstatePos:absorbingPos].reshape(nSamples, -1)
            absorbing = sast[:, absorbingPos]

            if self.scaled and self.iteration == 0:
                # create scaler and fit it
                self._sa_scaler = preprocessing.StandardScaler()
                sa[:, :-1] = self._sa_scaler.fit_transform(sa[:, :-1])

            if self.features is not None:
                sa = self.features(sa)
            # todo: non serve creare le features anche per lo stato prossimo?

            self.sa = sa
            self.snext = snext
            self.absorbing = absorbing
            self.r = r

    def _partial_fit(self, sast=None, r=None, **kwargs):
        """
        Perform a step of FQI using input data sast and r.
        Note that if the dataset does not change between iterations, you can
        provide None inputs after the first iteration. This speed up the computation
        because no normalization and preprocessing is performed and the initial data
        are used. However, if you want to change the dataset (e.g., for experience
        replay) you just need to provide a different dataset (sast, r) in input
        and everything works. The iteration exploits the current state of the estimator
        but the new dataset (after preprocessing).

        Args:
            sast (numpy.array, None): the input in the dataset
            r (numpy.array, None): the output in the dataset
            **kwargs: additional parameters to be provided to the fit function
            of the estimator

        Returns:
            sa, y: the preprocessed input and output
        """
        if self.iteration == 0 and (sast is None or r is None):
            raise ValueError('In the first iteration sast and r must be provided.')

        # normalize and store current data
        self._preprocessData(sast, r)

        # check if the estimator change the structure at each iteration
        adaptive = False

        # TODO: didn't understand what is models
        if hasattr(self.estimator, "models"):
            if hasattr(self.estimator.models[0], 'adapt'):
                adaptive = True
        else:
            if hasattr(self.estimator, 'adapt'):
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
         Note that it stores the state, action, nextstate and absorbing flag,
         this means that nfeatures = state_dim*2 + action_dim + 1
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
            states (numpy.array): states to be evaluated. Dimenions: (nsamples x state_dim)
            absorbing (bool): true if the current state is absorbing. Dimensions: (nsamples x 1)
        Returns:
            Q: the maximum Q-value in each state
            A: the action associated to the max Q-value in each state
        """
        newstate = self._checkStates(states, copy=True)
        nbstates = newstate.shape[0]

        if absorbing is None:
            absorbing = np.zeros((nbstates,))

        Q = np.zeros((nbstates, self._actions.shape[0]))
        for idx in range(self._actions.shape[0]):
            a = self._actions[idx, :].reshape(1, self.actionDim)
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

            assert np.allclose(Q[:, idx],
                               self._evaluate_Q(newstate, actions, absorbing).reshape(-1,1)),\
                'error in the function _evaluate_Q' # TODO check that it is correct
            #if it is correct the lines above can be replaced with
            # Q[:, idx] = self._evaluate_Q(newstate, actions, absorbing)

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
    
    # def predict(self, states, absorbing=None):
    #     """
    #     Compute the action with the highest Q value.
    #     Args:
    #         states (numpy.array): the states to be evaluated. Dimensions: (nsamples x state_dim)
    #         absorbing (bool,None): true if the current state is absorbing. Dimensions: (nsamples x 1)
    #     Returns:
    #         the argmax and the max Q value
    #
    #     """
    #     if self.iteration == 0:
    #         raise ValueError(
    #             'The model must be trained before being evaluated')
    #
    #     maxQ, maxa = self.maxQA(states, absorbing)
    #     return maxa, maxQ

    def drawAction(self, states, absorbing=None):
        """
        Compute the action with the highest Q value.
        Args:
            states (numpy.array): the states to be evaluated. Dimensions: (nsamples x state_dim)
            absorbing (bool,None): true if the current state is absorbing. Dimensions: (nsamples x 1)
        Returns:
            the argmax and the max Q value

        """
        if self.iteration == 0:
            raise ValueError(
                'The model must be trained before being evaluated')

        maxQ, maxa = self.maxQA(states, absorbing)
        return maxa

    def _evaluate_Q(self, states, actions, absorbing=None):
        """
        Evaluate the Q-function approximation in the given (state, action)-pair
        Args:
            states (numpy.array): the states to be evaluated. Dimensions: (nsamples x state_dim)
            actions (numpy.array): the actions to be evaluated. Dimensions: (nsamples x action_dim)
            absorbing (bool,None): true if the current state is absorbing. Dimensions: (nsamples x 1)
        Returns:
            the predicted Q-values
        """
        if self.iteration == 0:
            raise ValueError(
                'The model must be trained before being evaluated')

        newstate = self._checkStates(states, copy=True)
        newactions = self._checkActions(actions, copy=True)

        if absorbing is None:
            absorbing = np.zeros((newstate.shape[0],))

        # concatenate [newstate, action] and scalarize them
        if self.scaled:
            samples = np.concatenate((self._sa_scaler.transform(newstate),
                                      newactions), axis=1)
        else:
            samples = np.concatenate((newstate, newactions), axis=1)

        if self.features is not None:
            samples = self.features.testFeatures(samples)

        # predict Q-function
        predictions = self.estimator.predict(samples).ravel() * (1 - absorbing)

        return predictions

    def reset(self):
        """
        Reset FQI.

        """
        self.iteration = 0
        self.sa = None
        self.snext = None
        self.absorbing = None
        # TODO: reset estimator??
