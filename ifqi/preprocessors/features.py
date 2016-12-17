from sklearn.preprocessing import PolynomialFeatures


def select_features(f):
    if f is None:
        return None
    elif f['name'] == 'poly':
        return PolyFeatures(f['degree'])
    else:
        raise ValueError('unknown feature type.')


class PolyFeatures(object):
    def __init__(self, degree):
        self.degree = degree
        self.poly = PolynomialFeatures(self.degree)

    def __call__(self, X):
        return self.poly.fit_transform(X)

    def test_features(self, x):
        return self.poly.transform(x)
