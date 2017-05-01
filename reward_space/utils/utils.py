import numpy as np
from numpy.polynomial.chebyshev import chebvander, chebval
from numpy.polynomial.legendre import legval
import numpy.linalg as la

def kullback_leibler_divergence(pi1, pi2, tol=1e-24):
    return np.sum(pi1 * np.log(pi1 / (pi2 + tol) + tol)) / pi1.shape[0]

def exponentiated_gradient_decent(df, x0, lrate=0.1, max_iter=100, tol=1e-6):
    ite = 0
    x = x0
    grad = df(ite)
    while ite < max_iter and la.norm(grad) > tol:
        ite += 1
        grad = df(ite)
        x = x * np.exp(-lrate * grad)
    return x



def chebvanderNd(samples,n_dim,deg):
    '''
    Args:
        samples array/list of n_samples arrays of n_dim elements, each representing
                a sample
        n_dim   number of dimension in the samples
        deg     array/list n_dim elements, each representing the max degree
                of the polynomial
    
    Returns:
         a matrix of dimensions n_sample x (deg[0] + 1) x ... x (deg[n_dim-1] + 1)
         In each entry the product of Chebichev polynomials of the corresponding
         degree
    '''
    
    n_samples = len(samples)
    res = []
    vander_per_dim = []
    for j in range(n_dim):
        vander_per_dim.append(chebvander(samples[:,j],deg[j]))
    
    for i in range(n_samples):
        acc = vander_per_dim[0][i]
        for j in range(1,n_dim):
            acc = np.tensordot(acc,vander_per_dim[j][i],axes=0)
        res.append(acc)
                
    return np.array(res)

def chebvalNd(x,deg):
    '''
    Computes the n dimentional chebichev polynomial where each factor i
    has degree deg[i]
    '''
    max_deg = np.array(deg).max() + 1
    c = np.zeros(max_deg)
    prod = 1
    for i,d in enumerate(deg):
        c[d] = 1
        prod *= legval(x[i],c)
        c[d] = 0
    return prod

def legvalNd(x,deg):
    '''
    Computes the n dimentional legendre polynomial where each factor i
    has degree deg[i]
    '''
    max_deg = np.array(deg).max() + 1
    c = np.zeros(max_deg)
    prod = 1
    for i,d in enumerate(deg):
        c[d] = 1
        prod *= chebval(x[i],c)
        c[d] = 0
    return prod

class MinMaxScaler(object):
    '''
    Applies linear transformation from [x1,x2] to [y1,y2] s.t.
    y = m*x + q where m = (y1-y2)/(x1-x2) and q = (x1y2-x2y1)/(x1-x2)
    '''
    
    def __init__(self, ndim, input_ranges, output_ranges=None):
        if output_ranges is None:
            self.output_ranges = np.array([[-1,1]]*ndim, dtype=np.float64, ndmin=2)
        else:
            self.output_ranges = np.array(output_ranges, dtype=np.float64, ndmin=2)
        self.input_ranges = np.array(input_ranges, dtype=np.float64, ndmin=2)
        self.ndim = ndim
        self.m = np.array(map(lambda x,y: (y[0]-y[1])/(x[0]-x[1]), self.input_ranges, self.output_ranges))
        self.q = np.array(map(lambda x,y: (x[0]*y[1]-x[1]*y[0])/(x[0]-x[1]), self.input_ranges, self.output_ranges))
    
    def scale(self,data):
        data = np.array(data)
        return np.apply_along_axis(lambda x: self.m*x+self.q, 1, data)
        
class EpisodeIterator:
    def __init__(self, dataset, endofepisode_idx):
        self.dataset = dataset
        self.endofepisode_idx = endofepisode_idx
        self.n_episodes = dataset.shape[0]
        self.start_episode = 0
        self.end_episode = self.__find_episode_end()

    def __iter__(self):
        return self
    
    def __find_episode_end(self):
        for i in range(self.start_episode,self.n_episodes):
            if self.dataset[i][self.endofepisode_idx] == 1:
                return i + 1
        return self.n_episodes

    def next(self):
        if self.start_episode == self.n_episodes:
            raise StopIteration
        else:
            curr_episode = self.dataset[self.start_episode:self.end_episode]
            self.start_episode = self.end_episode
            self.end_episode = self.__find_episode_end()
            return curr_episode


def compute_feature_matrix(n_samples, n_features, states, actions, features):
    '''
    Computes the feature matrix X starting from the sampled data and
    the feature functions

    :param n_samples: number of samples
    :param n_features: number of features
    :param states: the states encountered in the run
    :param actions: the actions performed in the run
    :param features: a list of functions, each one is a feature
    :return: X the feature matrix n_samples x n_features
    '''
    X = np.zeros(shape=(n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_features):
            X[i, j] = features[j]([states[i], actions[i]])
    return X


def remove_projections(X, C, W):
    '''
    Makes the columns of matrix X orthogonal to the columns of
    matrix C, based on the weighted inner product with weights W
    '''
    P_cx = la.multi_dot([C.T, W, X])
    P_cc = la.multi_dot([C.T, W, C])
    C_norms2 = np.diag(np.diag(P_cc))
    P_cx_n = (np.power(C_norms2, -1)).dot(P_cx)
    X_ort = X - C.dot(P_cx_n)
    return X_ort


def find_basis(X, w):
    '''
    Finds an orthonormal basis for the space of the columns of matrix X
    based on the weighted inner product with weights w
    '''

    W = np.diag(w)
    W_inv = np.diag(np.power(w, -1))

    X_tilda_ort = np.sqrt(W).dot(X)
    U_ort, s_ort, V_ort = la.svd(X_tilda_ort)
    tol = s_ort.max() * max(X_tilda_ort.shape) * np.finfo(
        s_ort.dtype).eps  # as done in numpy
    U_tilda_ort_ort = U_ort[:, :s_ort.shape[0]][:, s_ort > tol]
    U_ort_ort = np.sqrt(W_inv).dot(U_tilda_ort_ort)
    return U_ort_ort
            
            
            