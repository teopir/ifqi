from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import rankdata
from numpy.matlib import repmat
import numpy as np

class Linear():
    def __init__(self,
                 nActions,
                 degree=3):
        self.nActions = nActions
        self.degree = degree
        self.poly = PolynomialFeatures(self.degree)
        self.X = None
        self.lastTestX = None
        self.lastTestSA = None
        self.availableActions = None
        self.model = self.initModel()
        
    def fit(self, X, y, **kwargs):
        if self.X is None:
            features = self.poly.fit_transform(X[:, :-1])
            self.availableActions = np.unique(X[:, -1]).round(4)
            actionIds = rankdata(X[:, -1], method='dense') - 1

            assert self.nActions == self.availableActions.size, \
                   'The dataset does not contain all the actions available \
                   in this environment.'
            
            rows = repmat(np.linspace(0, features.shape[0] - 1, features.shape[0]),
                          features.shape[1],
                          1
                          ).T.astype('int')
            cols = (actionIds * features.shape[1]).reshape(-1, 1)
            cols = repmat(cols, 1, features.shape[1])
            cols = cols + repmat(np.linspace(0,
                                             features.shape[1] - 1,
                                             features.shape[1]),
                                 features.shape[0],
                                 1).astype('int')
            features = repmat(features, 1, self.nActions)
            self.X = np.zeros((features.shape[0], features.shape[1]))
            self.X[rows, cols] += 1
            self.X *= features

        return self.model.fit(self.X, y, **kwargs)

    def predict(self, x, **kwargs):
        actionId = np.argwhere(x[0, -1].round(4) == self.availableActions).ravel()
        features = self.poly.transform(x[:, :-1])
        rows = repmat(np.linspace(0, features.shape[0] - 1, features.shape[0]),
                      features.shape[1],
                      1
                      ).T.astype('int')
        cols = (actionId * features.shape[1]).reshape(-1, 1)
        cols = repmat(cols, 1, features.shape[1])
        cols = cols + repmat(np.linspace(0,
                                         features.shape[1] - 1,
                                         features.shape[1]),
                             features.shape[0],
                             1).astype('int')
        features = repmat(features, 1, self.nActions)
        sa = np.zeros((features.shape[0], features.shape[1]))
        sa[rows, cols] += 1
        sa *= features
        
        return self.model.predict(sa, **kwargs)
        
    def adapt(self, iteration=1):
        pass

    def initModel(self):
        model = LinearRegression()

        return model