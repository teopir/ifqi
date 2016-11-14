import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

from ifqi.models.mlp import MLP


class Ensemble(object):
    def __init__(self):
        self._models = self.init_model()
        self._sum = np.zeros(y.shape)
        self._predict_sum = np.zeros(y.shape)

    def fit(self, X, y, **kwargs):
        delta = y - self._sum
        self._models[-1].fit(X, delta, **kwargs)
        self._sum += self.models[-1].predict(X).ravel()

    def predict(self, x, **kwargs):
        n_samples = x.shape[0]

        prediction = np.array((n_samples))
        for model in self.models:
            prediction += model.predict(x).ravel()

        return prediction

    def adapt(self, iteration):
        self.models.append(self.generate_model(iteration))

    def _init_model(self):
        model = self.generate_model(0)

        return [model]


class ExtraTreeEnsemble(Ensemble):
    def __init__(self,
                 n_estimators,
                 criterion,
                 min_samples_split,
                 min_samples_leaf):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        super(ExtraTreeEnsemble, self).__init__()

    def _generate_model(self, iteration):
        model = ExtraTreesRegressor(n_estimators=self.nEstimators,
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
                    self.hidden_neurons[iteration],
                    self.activation,
                    self.optimizer,
                    self.regularizer=None)

        return model
