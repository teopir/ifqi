from sklearn.preprocessing import PolynomialFeatures
import numpy as np

import ifqi.preprocessors.action_features as af


def select_features(f):
    if f is None:
        return None
    elif f['name'] == 'poly':
        phi = PolyFeatures
        params = f['params']
    else:
        raise ValueError('unknown feature type.')

    if 'action_phi' in f:
        if f['action_phi'] == 'andcondition':
            action_phi = af.AndCondition()
        else:
            raise ValueError('unknown basis function type.')
    else:
        action_phi = af.Identity()

    return phi(action_phi, **params)


class PolyFeatures(object):
    def __init__(self, action_phi, degree):
        self.action_phi = action_phi
        self.degree = degree
        self.poly = PolynomialFeatures(self.degree)

    def fit_transform(self, X):
        # TODO: currently this works only with action_dim = 1
        phi = self.poly.fit_transform(X[:, :-1])
        phi_a = np.concatenate((phi, X[:, -1:]), axis=1)

        discrete_actions = np.unique(X[:, -1])

        return self.action_phi.fit_transform(phi_a, discrete_actions)

    def transform(self, x):
        phi = self.poly.transform(x[:, :-1])
        phi_a = np.concatenate((phi, x[:, -1:]), axis=1)

        return self.action_phi.transform(phi_a)
