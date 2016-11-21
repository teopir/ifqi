import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

from ifqi.models.mlp import MLP

"""
Ensemble regressor.
This class is supposed to be used in this package and it may not work
properly outside.
"""


class Ensemble(object):
    def __init__(self):
        self._models = self._init_model()

    def fit(self, X, y, **kwargs):
        if not hasattr(self, '_target_sum'):
            self._target_sum = np.zeros(y.shape)
        delta = y - self._target_sum
        self._models[-1].fit(X, delta, **kwargs)
        self._target_sum += self._models[-1].predict(X).ravel()

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

    def _init_model(self):
        model = self._generate_model(0)

        return [model]


class ExtraTreesEnsemble(Ensemble):
    def __init__(self,
                 n_estimators,
                 criterion,
                 min_samples_split,
                 min_samples_leaf):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        super(ExtraTreesEnsemble, self).__init__()

    def _generate_model(self, iteration):
        model = ExtraTreesRegressor(n_estimators=self.n_estimators,
                                    criterion=self.criterion,
                                    min_samples_split=self.min_samples_split,
                                    min_samples_leaf=self.min_samples_leaf)

        return model


class MLPEnsemble(Ensemble):
    def __init__(self,
                 n_input,
                 n_output,
                 hidden_neurons,
                 activation,
                 optimizer,
                 regularizer=None):
        assert isinstance(hidden_neurons, list), 'hidden_neurons should be \
            of type list specifying the number of hidden neurons for each \
            hidden layer.'
        self.hidden_neurons = hidden_neurons
        self.optimizer = optimizer
        self.n_input = n_input
        self.n_output = n_output
        self.activation = activation
        self.regularizer = regularizer
        self.model = self.init_model()
        super(MLPEnsemble, self).__init__()

    def _generate_model(self, iteration):
        model = MLP(self.n_input,
                    self.n_output,
                    self.hidden_neurons,
                    self.activation,
                    self.optimizer,
                    self.regularizer)

        return model
