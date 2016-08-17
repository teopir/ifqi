from scipy.stats import rankdata
from numpy.matlib import repmat
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def selectFeatures(f):
    if f['name'] == 'poly':
        return PolyFeatures(f['degree'])
    else:
        print('Unknown feature type. None will be applied')
        return None

class PolyFeatures(object):
    def __init__(self, degree):
        self.degree = degree
        self.poly = PolynomialFeatures(self.degree)
        self.availableActions = None

    def extractFeaturesFit(self, X):
        features = self.poly.fit_transform(X[:, :-1])
        self.availableActions = np.unique(X[:, -1]).round(4)
        actionIds = rankdata(X[:, -1], method='dense') - 1
    
        sa = self._computeFeatures(features, actionIds)
        
        return sa
        
    def extractFeaturesPredict(self, x):
        features = self.poly.transform(x[:, :-1])
        actionId = np.argwhere(x[0, -1].round(4) == self.availableActions)
        if len(actionId) == 0:
            raise ValueError('Action to predict not seen in the training phase')
    
        sa = self._computeFeatures(features, actionId)
        
        return sa
        
    def _computeFeatures(self, features, actionIds):
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
        features = repmat(features, 1, self.availableActions.size)
        sa = np.zeros((features.shape[0], features.shape[1]))
        sa[rows, cols] += 1
        sa *= features
        
        return sa