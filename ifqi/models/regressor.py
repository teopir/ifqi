import sklearn.preprocessing as preprocessing
import time
import numpy as np

class Regressor:

    def __init__(self,regressor_class=None,input_scaled=True, output_scaled=True,**kwargs):
        self.regressor = regressor_class(**kwargs)
        self._input_scaled = input_scaled
        self._output_scaled = output_scaled

    def fit(self, X, y, **kwargs):

        if self._input_scaled:
            self._pre_X = preprocessing.StandardScaler()
            X = self._pre_X.fit_transform(X)


        if self._output_scaled:
            self._pre_y = preprocessing.StandardScaler()
            y = np.reshape(y,(-1,1))
            y = self._pre_y.fit_transform(y).ravel()


        return self.regressor.fit(X,y, **kwargs)

    def predict(self, X, **kwargs):

        if self._input_scaled:
            X = self._pre_X.transform(X)

        y = self.regressor.predict(X, **kwargs)

        if self._output_scaled:
            y = np.reshape(y, (-1, 1))
            y = self._pre_y.inverse_transform(y).ravel()

        return y