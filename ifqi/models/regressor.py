import sklearn.preprocessing as preprocessing

from ifqi.preprocessors.features import select_features


class Regressor(object):
    def __init__(self, regressor_class=None, **kwargs):
        self.features = select_features(kwargs.pop('features', None))
        self._input_scaled = kwargs.pop('input_scaled', None)
        self._output_scaled = kwargs.pop('output_scaled', None)

        self._regressor = regressor_class(**kwargs)

    def fit(self, X, y, **kwargs):
        if self.features:
            X = self.features.fit_transform(X)

        if self._input_scaled:
            self._pre_X = preprocessing.StandardScaler()
            X = self._pre_X.fit_transform(X)

        if self._output_scaled:
            self._pre_y = preprocessing.StandardScaler()
            y = self._pre_y.fit_transform(y).ravel()

        return self._regressor.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        if self.features:
            X = self.features.transform(X)

        if self._input_scaled:
            if hasattr(self, '_pre_X'):
                X = self._pre_X.transform(X)
            else:
                X = preprocessing.StandardScaler().fit_transform(X)

        y = self._regressor.predict(X, **kwargs)
        if self._output_scaled:
            y = self._pre_y.inverse_transform(y).ravel()

        return y

    def get_weights(self):
        return self._regressor.get_weights()

    def set_weights(self, w):
        return self._regressor.set_weights(w)

    def count_params(self):
        return self._regressor.count_params()

    @property
    def layers(self):
        if hasattr(self._regressor, 'layers'):
            return self._regressor.layers
        else:
            None
