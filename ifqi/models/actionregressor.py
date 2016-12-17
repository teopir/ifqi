from builtins import range
from copy import deepcopy

import numpy as np

from ifqi.models.ensemble import Ensemble


class ActionRegressor(object):
    """
    This class is a meta-regressor to be used when the actions are discrete.
    It stores an independent regressor for each discrete action.
    This is useful when discrete actions are used because the action space
    in this scenario may not be a metric space so it is not necessary to
    exploit spatial correlation along action space.
    """

    def __init__(self, model, discrete_actions, decimals, **params):
        """
        Initialization of the class.

        Parameters:
            model (object): an estimator
            discrete_actions (int, list): when an integer is given it
                represents the number of discrete actions to be used
                [0, 1, 2, discrete_actions - 1]. Otherwise the values
                contained in the list are used.
            decimals (int): precision for float actions
            **params: additional parameters that are used to init the model
        """
        if isinstance(discrete_actions, (int, float)):
            discrete_actions = np.arange(int(discrete_actions))
            self._decimals = 0
        else:
            # fix number of decimals (i.e., precision)
            discrete_actions = np.around(discrete_actions, decimals=decimals)
            self._decimals = decimals

        # transform discrete actions into a matrix
        dim = len(discrete_actions.shape)
        if dim == 1:
            discrete_actions = discrete_actions.reshape(-1, 1)
        elif dim > 2:
            raise ValueError('Such dimensionality cannot be handled')

        # remove duplicated actions
        b = np.ascontiguousarray(discrete_actions).view(np.dtype(
            (np.void,
             discrete_actions.dtype.itemsize * discrete_actions.shape[1])))
        self._actions = np.unique(b).view(
            discrete_actions.dtype).reshape(-1, discrete_actions.shape[1])
        # actions is a #action x #variables. Ie each row is an action
        if self._decimals == 0:
            self._actions = self._actions.astype('int')
        self._actions = np.sort(self._actions.ravel())

        self._models = self._init_model(model, **params)

    def fit(self, X, y, **kwargs):
        """
        Split the input data according to the contained action. Each new set
        is used to fit the associated model.
        Parameters:
            X (np.array): Training data. Last column must contain the action
                          (it is used as splitting criteria).
                          Dimensions: n_samplex x n_features
            y (np.array): Target values. Dimensions: n_samples x 1
            **kwargs: additional parameters to be passed to the fit function of
                      the estimator
        """
        for i in range(len(self._models)):
            action = self._actions[i]
            idxs = np.all(X[:, -1:] == action, axis=1)

            self._models[i].fit(X[idxs, :-1], y[idxs], **kwargs)

    def predict(self, x, **kwargs):
        """
        Predict the target for sample x using the estimator associated to
        the action contained in x. Action must be stored in the last column of
        x.

        Parameters:
            x (np.array): Test point. Last column must contain the action
                          (it is used to select the estimator).
                          Dimensions: 1 x n_features
            **kwargs: additional parameters to be passed to the
                      predict function of the estimator

        Returns:
            output (np.array): target associated to sample x
        """

        predictions = np.zeros(x.shape[0])
        for i in range(self._actions.shape[0]):
            action = self._actions[i]
            idxs = np.all(x[:, -1:] == action, axis=1)

            if np.any(idxs):
                p = self._models[i].predict(x[idxs, :-1], **kwargs)
                predictions[idxs] = p

        return predictions

    def adapt(self, iteration):
        if hasattr(self._models[0], 'adapt'):
            for model in self._models:
                model.adapt(iteration)

    def _init_model(self, model, **params):
        """
        Initialize a new estimator for each discrete action.
        The output is a list of estimators with length equal to the
        number of discrete actions.

        Parameters:
            model (object): an instance of estimator
            **params: additional parameters to be passed to the constructor
        Returns:
            models (list): list of initialized estimators
        """
        models = list()
        for i in range(self._actions.shape[0]):
            models.append(deepcopy(model))

        return models
