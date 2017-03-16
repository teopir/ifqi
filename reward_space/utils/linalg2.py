import numpy as np
import numpy.linalg as la
import statsmodels.api as sm

def nullspace(A, criterion='rank', atol=1e-13, rtol=0):

    '''
    Computes the null space of matrix A
    :param A: the matrix
    :param criterion: 'rank' or 'tol' If 'rank' it uses the rank of matrix A
                      to determine the rank of the null space, otherwise it uses
                      the tolerance
    :param atol:    absolute tolerance
    :param rtol:    relative tolerance
    :return:        the matrix whose columns are the null space of A
    '''

    A = np.atleast_2d(A)
    u, s, vh = la.svd(A)
    if criterion == 'tol':
        tol = max(atol, rtol * s[0])
        nnz = (s >= tol).sum()
    else:
        nnz = la.matrix_rank(A)
    ns = vh[nnz:].conj().T
    return ns

def range(A, criterion='rank', atol=1e-13, rtol=0):
    """Compute an approximate basis for the range of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        range of `A`.  The columns of `ns` are a basis for the
        range.
    """

    A = np.atleast_2d(A)
    u, s, vh = la.svd(A)
    if criterion == 'tol':
        tol = max(atol, rtol * s[0])
        nnz = (s >= tol).sum()
    else:
        nnz = la.matrix_rank(A)
    ns = u[:,:nnz]
    return ns

def lsq(X, y, w=None):
    if w is None:
        w, _, _, _ = la.lstsq(X, y)
        y_hat = X.dot(w)
        rmse = np.sqrt(la.norm(y - y_hat) / X.shape[0])
        mae = la.norm(y - y_hat, ord = 1) / X.shape[0]
    else:
        mod_wls = sm.WLS(y, X, weights=1. / w)
        res_wls = mod_wls.fit()
        w = res_wls.params
        y_hat = res_wls.fittedvalues
        rmse = np.sqrt(np.dot(w, (y - y_hat)**2) / np.sum(w))
        mae = np.dot(w, abs(y - y_hat)) / np.sum(w)

    return y_hat, w, rmse, mae

class InnerProductSpace(object):

    def __init__(self, dim, measure=None, inner_product=None):
        self.dim = dim

        if measure is None:
            self.measure = np.ones(dim)
        else:
            self.measure = measure
        self.diag_measure = np.diag(self.measure)

        if inner_product is None:
            self.inner_product = lambda x,y: la.multi_dot(x.transpose(),
                                                          self.diag_measure, y)
        else:
            self.inner_product = inner_product

    def dot(self,x,y):
        return self.inner_product(x,y)

    def norm(self,x):
        return self.dot(x,x)

    def nullspace(self,X):
        pass

    def range(self,X):
        pass

    def rank(self,X):
        pass

    def gram_shmitt(self,X):
        pass

    def ls(self, X, y):
        mod_wls = sm.WLS(y, X, weights= 1. / (self.tol + self.measure))
        res_wls = mod_wls.fit()
        X_hat = res_wls.fittedvalues
        rmse = np.sqrt(res_wls.ssr)
        w = res_wls.params
        return X_hat, w, rmse



