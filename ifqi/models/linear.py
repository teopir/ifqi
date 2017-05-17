from sklearn import linear_model as lm

"""
Scikit-learn linear models wrapper.

"""


class Linear(object):
    def __init__(self):
        self.model = self.init_model()

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, x, **kwargs):
        predictions = self.model.predict(x, **kwargs)
        return predictions.ravel()

    def adapt(self, iteration):
        pass

    def init_model(self):
        return lm.LinearRegression()

    def count_params(self):
        if hasattr(self.model, 'coef_'):
            return self.model.coef_.size
        else:
            return 0

    def get_weights(self):
        if hasattr(self.model, 'coef_'):
            return self.model.coef_
        else:
            return None

    def set_weights(self, w):
        if hasattr(self.model, 'coef_'):
            self.model.coef_ = w


class Ridge(Linear):
    def __init__(self):
        super(Ridge, self).__init__()

    def init_model(self):
        return lm.Ridge()
