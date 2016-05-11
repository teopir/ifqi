from __future__ import print_function
import numpy as np
import sys
import numpy.matlib
import sklearn.preprocessing as preprocessing

class FQI:
    def __init__(self, estimator, stateDim, actionDim,
                 discrete_actions=10,
                 gamma=0.9, horizon=10,
                 scaled=False, verbose=0):
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
            self._actions = discrete_actions
        elif isinstance(discrete_actions, list):
            assert len(discrete_actions) > 1, 'Error: at least two actions are required'
            self._actions = np.array(discrete_actions, dtype='float32').reshape(-1, actionDim)
        elif isinstance(discrete_actions, int):
            assert discrete_actions > 1, 'Error: at least two actions are required'
        else:
            raise ValueError('Supported types for diacrete_actions are {np.darray, list, int}')

        self.__name__ = "FittedQIteration"
        self.iteration = 0
        self.scaled = scaled

    def _check_states(self, X, copy=True):
        return X.reshape(-1, self.stateDim)

    def _compute_actions(self, sast):
        action_po = self.stateDim
        nextstate_pos = self.stateDim + self.actionDim
        #select unique actions
        if isinstance(self.discrete_actions, int):
            actions = sast[:, action_po:nextstate_pos].reshape(-1, self.actionDim)
            ubound = np.amax(actions, axis=0)
            lbound = np.amin(actions, axis=0)
            if self.actionDim == 1:
                self._actions = np.linspace(lbound, ubound, self.discrete_actions).reshape(-1,1)
            else:
                print("not implemented in the general case (action_dim > 1")
                exit(9)

    def _preprocess_data(self, sast=None, r=None):

        if sast is not None and r is not None:
            # get number of samples
            nSamples = sast.shape[0]
            action_po = self.stateDim
            nextstate_pos = self.stateDim + self.actionDim
            absorbing_pos = nextstate_pos + self.stateDim

            sa = np.copy(sast[:, 0:nextstate_pos]).reshape(nSamples,-1)
            snext = sast[:, nextstate_pos:absorbing_pos].reshape(nSamples,-1)
            absorbing = sast[:, absorbing_pos]

            if self.scaled and self.iteration == 0:
                # create scaler and fit it
                self._sa_scaler = preprocessing.StandardScaler()
                sa = self._sa_scaler.fit_transform(sa)

            self.sa = sa
            self.snext = snext
            self.absorbing = absorbing
            self.r = r


    def _partial_fit(self, sast=None, r=None, **kwargs):
        # compute the action list
        if not hasattr(self, '_actions'):
            self._compute_actions(sast)

        self._preprocess_data(sast, r)

        #check if the estimator change the structure at each iteration
        adaptive = False
        if hasattr(self.estimator, 'adapt'):
            adaptive = True

        y = self.r
        if self.iteration == 0:
            if self.verbose > 0:
                print('Iteration {}'.format(self.iteration+1))
                
            self.estimator.fit(self.sa, self.r, **kwargs)
        else:
            maxq, maxa = self.maxQA(self.snext, self.absorbing)
            y = self.r + self.gamma * maxq

            if self.verbose > 0:
                print('Iteration {}'.format(self.iteration+1))

            if adaptive:
                # update estimator structure
                self.estimator.adapt(iteration=self.iteration)

            self.estimator.fit(self.sa, y, **kwargs)

        self.iteration += 1

        return self.sa, y

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
            print("Starting complete FQI ...")

        # reset iteration count
        self.reset()
        # compute/recompute the action set
        self._compute_actions(sast)

        # main loop
        self._partial_fit(sast, r)
        for t in range(1, self.horizon):
            self._partial_fit()

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
            a = self._actions[idx,:].reshape(1, self.actionDim)
            actions = np.matlib.repmat(a, nbstates, 1)

            # concatenate [newstate, action] and scalarize them
            if self.scaled:
                samples = self._sa_scaler.transform(
                    np.concatenate((newstate, actions), axis=1)
                )
            else:
                samples = np.concatenate((newstate, actions), axis=1)

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
        #sanity check
        #assert(np.allclose(rQ, Q[np.arange(nbstates), amax]))

        return rQ, rA

    def partial_fit(self, X, y, **kwargs):
        self._partial_fit(X, y, **kwargs)
        return self

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

    def predict(self, states):
        if not hasattr(self, '_actions'):
            raise ValueError('The model must be trained before to be evaluated')

        maxQ, maxa = self.maxQA(states)
        return maxa, maxQ

    def reset(self):
        self.iteration = 0
        self.sa = None
        self.snext = None
        self.absorbing = None


