import numpy as np
from builtins import range


class ActionRegressor(object):
    """
    This class is a meta-regressor to be used when the actions are discrete.
    It stores an independent regressor for each discrete action.
    This is useful when discrete actions are used because the action space
    in this scenario may not be a metric space so it is not necessary to exploit
    spatial correlation along action space.
    """

    def __init__(self, model, discreteActions, decimals=6, **params):
        """
        Initialization of the class.
        
        Parameters:
            model (object): an estimator
            discreteActions (int, list): when an integer is given it
                represents the number of discrete actions to be used
                [0, 1, 2, discreteActions-1]. Otherwise the values
                contained in the list are used.
            **params: additional parameters that are used to init the model
        """
        if isinstance(discreteActions, ('int', 'float')):
            discreteActions = np.arange(int(discreteActions))
            self.decimals = 0
        else:
            # fix number of decimals (i.e., precision)
            discreteActions = np.around(discreteActions, decimals=decimals)
            self.decimals = decimals
        self.actions = np.unique(discreteActions)
        if self.decimals == 0:
            self.actions = self.actions.astype('int')
        self.models = self.initModel(model, **params)

    def fit(self, X, y, **kwargs):
        """
        Split the input data according to the contained action. Each new set
        is used to fit the associated model.
        Parameters:
            X (np.array): Training data. Last column must contain the action (it is used
                          as splitting criteria). Dimensions: n_samplex x n_features
            y (np.array): Target values. Dimensions: n_samples x 1
            **kwargs: additional parameters to be passed to the fit function of the estimator
        Returns:
            None
        """
        # todo arange on X[:, -1]
        for i in range(len(self.models)):
            action = self.actions[i]
            idxs = np.argwhere(X[:, -1] == action).ravel()
            self.models[i].fit(X[idxs, :-1], y[idxs], **kwargs)

    def predict(self, x, **kwargs):
        """
        Predict the target for sample x using the estimator associated to
        the action contained in x. Action must be stored in the last column of x

        Parameters:
            x (np.array): Test point. Last column must contain the action (it is used
                          to select the estimator). Dimensions: 1 x n_features
            **kwargs: additional parameters to be passed to the predict function
                      of the estimator

        Returns:
            output (np.array): target associated to sample x
        """
        assert x.shape[0] == 1
        action = x[0, -1]
        idxs = np.asscalar(np.argwhere(self.actions == action))
        output = self.models[idxs].predict(x[:, :-1], **kwargs)
        return output

    def adapt(self, iteration):
        for i in range(len(self.models)):
            self.models[i].adapt(iteration)

    def initModel(self, model, **params):
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
        for i in range(len(self.actions)):
            models.append(model(**params))
        return models
