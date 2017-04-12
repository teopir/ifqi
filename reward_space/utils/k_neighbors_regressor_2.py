from sklearn.neighbors.base import _get_weights
from sklearn.utils import check_array
from sklearn.neighbors import KNeighborsRegressor
import numpy as np


class KNeighborsRegressor2(KNeighborsRegressor):

    def __init__(self, n_neighbors=5, weights='uniform',
                 algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=1,
                 **kwargs):
        super(KNeighborsRegressor2, self).__init__(n_neighbors, weights,
                 algorithm, leaf_size, p, metric, metric_params, n_jobs,
                 **kwargs)

    def predict(self, X, rescale=True):
        """Predict the target for the provided data
        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.
        rescale: whether to rescale the prediction over the weights
        Returns
        -------
        y : array of int, shape = [n_samples] or [n_samples, n_outputs]
            Target values
        """
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            y_pred = np.mean(_y[neigh_ind], axis=1)
        else:
            y_pred = np.empty((X.shape[0], _y.shape[1]), dtype=np.float64)
            denom = np.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                if rescale:
                    y_pred[:, j] = num / denom
                else:
                    y_pred[:, j] = num

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred