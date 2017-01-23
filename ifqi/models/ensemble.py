import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import ExtraTreesRegressor
from ifqi.models.regressor import Regressor
from sklearn.linear_model import LinearRegression

from ifqi.models.mlp import MLP

"""
Ensemble regressor.
This class is supposed to be used in this package and it may not work
properly outside.
"""


class Ensemble(object):

    def __init__(self, ens_regressor_class=None, **kwargs):
        self._callings_args = kwargs
        self._regressor_class = ens_regressor_class
        self._models = self._init_model()

    def fit(self, X, y, **kwargs):
        if not hasattr(self, '_target_sum'):
            self._target_sum = np.zeros(y.shape)
        delta = y - self._target_sum
        ret = self._models[-1].fit(X, delta, **kwargs)
        self._target_sum += self._models[-1].predict(X).ravel()
        return ret

    def predict(self, x, **kwargs):
        if 'idx' in kwargs:
            idx = kwargs['idx']
            n_actions = kwargs['n_actions']
            if not hasattr(self, '_predict_sum'):
                self._predict_sum = np.zeros((x.shape[0], n_actions))

            predictions = self._models[-1].predict(x).ravel()
            self._predict_sum[:, idx] += predictions

            return self._predict_sum[:, idx]

        prediction = np.zeros(x.shape[0])
        for model in self._models:
            prediction += model.predict(x).ravel()

        return prediction

    def adapt(self, iteration):
        self._models.append(self._generate_model(iteration))

    def has_ensembles(self):
        return True

    def reset(self):
        pass

    def _init_model(self):
        model = self._generate_model(0)
        return [model]

    def _generate_model(self, iteration):
        return self._regressor_class(**self._callings_args)