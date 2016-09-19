import numpy as np
from copy import deepcopy


class ActionRegressor(object):
    def __init__(self, model, nActions, **params):
        self.nActions = nActions
        self.models = self.initModel(model, **params)

    def fit(self, X, y, **kwargs):
        for i in range(len(self.models)):
            idxs = np.argwhere(X[:, -1] == i).ravel()
            self.models[i].fit(X[idxs, :-1], y[idxs], **kwargs)

    def predict(self, x, **kwargs):
        i = x[0, -1].astype('int')
        output = self.models[i].predict(x[:, :-1], **kwargs)

        return output

    def adapt(self, iteration):
        for i in range(len(self.models)):
            self.models[i].adapt(iteration)

    def initModel(self, model, **params):
        models = list()
        for i in range(self.nActions):
            models.append(model(**params))
        return models
