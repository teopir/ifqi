from sklearn.preprocessing import PolynomialFeatures


def select_features(f):
    if f is None:
        return None
    elif f['name'] == 'poly':
        phi = PolyFeatures(f['params']['degree'])
    else:
        raise ValueError('unknown feature type.')

    return phi


class PolyFeatures(object):
    def __init__(self, degree):
        self.degree = degree
        self.poly = PolynomialFeatures(self.degree)

    def fit_transform(self, X):
        return self.poly.fit_transform(X[:, :])

    def transform(self, x):
        return self.poly.transform(x[:, :])
