from __future__ import print_function
import numpy as np
import sys
import numpy.matlib

class FQI:
    def __init__(self, estimator, stateDim, actionDim,
                 discrete_actions=10,
                 gamma=0.9, horizon=10, verbose=0):
        self.estimator = estimator
        self.gamma = gamma
        self.horizon = horizon

        self.stateDim = stateDim
        self.actionDim = actionDim

        self.verbose = verbose
        self.discrete_actions = discrete_actions
        if isinstance(discrete_actions, np.ndarray):
            assert discrete_actions.shape[1] == actionDim
            assert discrete_actions.shape[0] > 1
        else:
            assert discrete_actions > 1, 'Error: at least two actions are required'

        self.__name__ = "FittedQIteration"

    def _checkInputs(self, X, y):
        nFeatures = X.shape[1]
        assert nFeatures == self.stateDim + self.actionDim, \
            'state/action dimension do not match the dimension of the provided data'

        assert y.shape[1] == 1,\
            'reward must be a single value'

        assert X.shape[0] == y.shape[0],\
            'X and y must have the same number of samples'

    def _check_states(self, X, copy=True):
        return X.reshape(-1, self.stateDim)

    def _fit(self, sast, r, **kwargs):
        """

        :param sast: array-like, [nbsamples x nfeatures]
         Note that it stores the state, action, nextstate and absorbing flag, this means that
         nfeatures = state_dim*2 + action_dim + 1
        :param r: array-like, [nbsamples x 1]
         Note it stores the reward function associated to the transition sas
        :return: None
        """
        if self.verbose > 0:
            print("Starting FQI ...")

        # get number of samples
        nSamples = sast.shape[0]
        action_po = self.stateDim
        nextstate_pos = self.stateDim + self.actionDim
        absorbing_pos = nextstate_pos + self.stateDim

        sa = np.copy(sast[:, 0:nextstate_pos]).reshape(nSamples,-1)
        snext = sast[:, nextstate_pos:absorbing_pos].reshape(nSamples,-1)
        absorbing = sast[:, absorbing_pos].reshape(-1,1)

        #select unique actions
        if isinstance(self.discrete_actions, int):
            actions = sast[:, action_po:absorbing_pos].reshape(-1, self.actionDim)
            ubound = np.amax(actions, axis=0)
            lbound = np.amin(actions, axis=0)
            if self.actionDim == 1:
                self._actions = np.linspace(lbound, ubound, self.discrete_actions).reshape(-1,1)
            else:
                print("not implemented in the general case (action_dim > 1")
                exit(9)
        else:
            self._actions = self.discrete_actions

        #check if the estimator change the structure at each iteration
        adaptive = False
        if hasattr(self.estimator, 'adapt'):
            adaptive = True

        # fit reward
        if self.verbose > 0:
            print('Iteration 0: fitting reward')
        self.estimator.fit(sa, r, **kwargs)

        # main loop
        for t in range(self.horizon):

            maxq, maxa = self.maxQA(snext, absorbing)
            y = r + self.gamma * maxq

            if self.verbose > 0:
                print('Iteration {}'.format(t+1))

            if adaptive:
                # update estimator structure
                self.estimator.adapt(iteration=t)

            self.estimator.fit(sa, y, **kwargs)

    def maxQA(self, states, absorbing=None):
        """
        Computes the maximum Q-function and the associated action
        in the provided states

        :param states: array-like, [nbstate x state_dim]
        :param absorbing: array-like, [nbstate x 1]
            {0,1} array representing if the state is absorbing
        :return:
        """
        newstate = self._check_states(states, copy=True)
        nbstates = newstate.shape[0]

        if absorbing is None:
            absorbing = np.zeros((nbstates,))

        Q = np.zeros((nbstates, self._actions.shape[0]))
        for idx in range(self._actions.shape[0]):
            a = self._actions[idx,:].reshape(1,-1)
            actions = np.matlib.repmat(a, nbstates, 1)
            samples = np.concatenate((newstate, actions), axis=1)
            predictions = self.estimator.predict(samples)

            Q[:, idx] = predictions.ravel()
            Q[:, idx] = Q[:, idx] * (1 - absorbing).ravel()

        amax = np.argmax(Q, axis=1)
        return Q[np.arange(nbstates), amax], amax

    def fit(self, X, y, **kwargs):
        """
        :param sast: array-like, [nbsamples x nfeatures]
         Note that it stores the state, action, nextstate and absorbing flag, this means that
         nfeatures = state_dim*2 + action_dim + 1
        :param r: array-like, [nbsamples x 1]
         Note it stores the reward function associated to the transition sas
        :return:
        """
        self._fit(X, y, **kwargs)
        return self
