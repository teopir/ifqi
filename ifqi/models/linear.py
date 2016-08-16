from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

class Linear():
    def __init__(self,
                 degree=3):
        self.degree = degree
        self.poly = PolynomialFeatures(self.degree)
        self.X = None
        self.model = self.initModel()
        
    def fit(self, X, y, **kwargs):
        if self.X is None:
            self.X = self.poly.fit_transform(X[:, :-1])
            actions = np.copy(X[:, -1])
            actions = actions.reshape(-1, 1)
            self.X = np.concatenate((self.X, actions), axis=1)

        return self.model.fit(self.X, y, **kwargs)
      
    def predict(self, x, **kwargs):
        features = self.poly.transform(x[:, :-1])
        actions = np.copy(x[:, -1])
        actions = actions.reshape(-1, 1)
        stateAction = np.concatenate((features, actions), axis=1)
        
        return self.model.predict(stateAction, **kwargs)
        
    def adapt(self, iteration=1):
        pass

    def initModel(self):
        model = LinearRegression()

        return model