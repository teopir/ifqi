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

    def __init__(self, model, discreteActions, decimals, **params):
        """
        Initialization of the class.
        
        Parameters:
            model (object): an estimator
            discreteActions (int, list): when an integer is given it
                represents the number of discrete actions to be used
                [0, 1, 2, discreteActions-1]. Otherwise the values
                contained in the list are used.
            decimals (int): precision for float actions
            **params: additional parameters that are used to init the model
        """
        if isinstance(discreteActions, (int, float)):
            discreteActions = np.arange(int(discreteActions))
            self.decimals = 0
        else:
            # fix number of decimals (i.e., precision)
            discreteActions = np.around(discreteActions, decimals=decimals)
            self.decimals = decimals

        # transform discrete actions into a matrix
        dim = len(discreteActions.shape)
        if dim == 1:
            discreteActions = discreteActions.reshape(-1, 1)
        elif dim > 2:
            raise ValueError('Such dimensionality cannot be handled')

        # remove duplicated actions
        b = np.ascontiguousarray(discreteActions).view(np.dtype(
            (np.void, discreteActions.dtype.itemsize * discreteActions.shape[1])))
        self.actions = np.unique(b).view(discreteActions.dtype).reshape(-1, discreteActions.shape[1])
        # actions is a #action x #variables. Ie each row is an action
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
            #TODO: I should not write -1 but -actionDim
            idxs = np.all(X[:, -1:] == action, axis=1)
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

        ret = np.zeros((x.shape[0]))
        for idAction in range(self.actions.shape[0]):

            #select all xs that belong to action of idAction
            idxs = np.all(x[:, -1:] == self.actions[idAction], axis=1)
            #return...
            if(np.sum(idxs)>0):
                if (np.sum(idxs) < x.shape[0]):
                    print("Error Catch!")
                if (len(idxs.shape)>1):
                    n = idxs.shape[0]
                    idxs = np.reshape(np.asarray(idxs),(n))
                ret[idxs] = self.models[idAction].predict(x[idxs, :-1], **kwargs)

        return ret

    def adapt(self, iteration):
        for i in range(len(self.models)):
            if hasattr(self.models[i], "adapt"):
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
        for i in range(self.actions.shape[0]):
            models.append(model(**params))
        return models
