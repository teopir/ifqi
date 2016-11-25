import sklearn.preprocessing as preprocessing


class Regressor:
    def __init__(self, regressor_class=None, **kwargs):
        self._input_scaled = kwargs.pop('input_scaled', None)
        self._output_scaled = kwargs.pop('output_scaled', None)
        self._regressor = regressor_class(**kwargs)

    def fit(self, X, y, **kwargs):
        if self._input_scaled:
            self._pre_X = preprocessing.StandardScaler()
            X = self._pre_X.fit_transform(X)

        if self._output_scaled:
            self._pre_y = preprocessing.StandardScaler()
            y = self._pre_y.fit_transform(y.reshape(-1, 1)).ravel()

        return self._regressor.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        if self._input_scaled:
            X = self._pre_X.transform(X)

        y = self._regressor.predict(X, **kwargs)
        if self._output_scaled:
            y = self._pre_y.inverse_transform(y).ravel()

        return y
