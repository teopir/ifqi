import numpy as np
from joblib import Parallel, delayed

from ifqi.models.regressor import Regressor

"""
Ensemble regressor.
This class is supposed to be used in this package and it may not work
properly outside.
"""


def pred(x, m):
    return m.predict(x)


class Ensemble(object):
    def __init__(self, regressor_class=None, **kwargs):
        self._regressor_class = regressor_class
        self._regr_args = kwargs
        self._models = self._init_model()

    def fit(self, X, y, **kwargs):
        if not hasattr(self, '_target_sum'):
            self._target_sum = np.zeros(y.shape)
        delta = y - self._target_sum
        self._models[-1].fit(X, delta, **kwargs)
        self._target_sum += self._models[-1].predict(X).ravel()

    def predict(self, x, **kwargs):
        if 'action_idx' in kwargs:
            action_idx = kwargs['action_idx']
            n_actions = kwargs['n_actions']
            if not hasattr(self, '_predict_sum'):
                self._predict_sum = np.zeros((x.shape[0], n_actions))

            predictions = self._models[-1].predict(x).ravel()
            self._predict_sum[:, action_idx] += predictions

            return self._predict_sum[:, action_idx]

        prediction = np.zeros(x.shape[0])
        for model in self._models:
            prediction += model.predict(x).ravel()

        return prediction

    def adapt(self, iteration):
        self._models.append(self._generate_model(iteration))

    def _init_model(self):
        model = self._generate_model(0)

        return [model]

    def _generate_model(self, iteration):
        return Regressor(self._regressor_class, **self._regr_args)
