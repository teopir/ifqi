import sklearn.preprocessing as preprocessing
import time

class Regressor:

    def __init__(self,regressor_class=None,input_scaled=True, output_scaled=False,**kwargs):
        self.regressor = regressor_class(**kwargs)
        self._input_scaled = input_scaled
        self._output_scaled = output_scaled

    def fit(self, X, y, **kwargs):

        if self._input_scaled:
            now = time.time()
            self._pre_X = preprocessing.StandardScaler()
            X = self._pre_X.fit_transform(X)
            print("scaling time = " , time.time() - now)

        if self._output_scaled:
            self._pre_y = preprocessing.StandardScaler()
            y = self._pre_y.fit_transform(y).ravel()

        return self.regressor.fit(X,y, **kwargs)

    def predict(self, X, **kwargs):

        if self._input_scaled:
            X = self._pre_X.transform(X)

        y = self.regressor.predict(X, **kwargs)

        if self._output_scaled:
            y = self._pre_y.inverse_transform(y).ravel()

        return y