import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def select_features(f):
    if f is None:
        return None
    elif f['name'] == 'poly':
        return PolyFeatures(f['degree'])
    else:
        print('Unknown feature type. None will be applied')
        return None


class PolyFeatures(object):
    def __init__(self, degree):
        self.degree = degree
        self.poly = PolynomialFeatures(self.degree)

    def __call__(self, X):
        return np.concatenate((self.poly.fit_transform(X[:, :-1]),
                               X[:, -1].reshape(-1, 1)),
                              axis=1)

    def test_features(self, x):
        return np.concatenate((self.poly.transform(x[:, :-1]),
                               x[:, -1].reshape(-1, 1)),
                              axis=1)
