import numpy as np
from numpy.polynomial.chebyshev import chebvander
from numpy.polynomial.legendre import legvander
from utils import MinMaxScaler

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

def sample_norms(samples):
    square_samples = samples**2
    return np.sqrt(square_samples.sum(axis=0))

def sample_log_gradients(policy,features):
    return np.array(map(lambda v: policy.gradient_log_pdf(v[0],v[1]), features), ndmin=2)


class OrtogonalPolynomialSelector(object):
    
    def __init__(self, ndim, bounds, max_degree = None):
        if max_degree == None:
            self.max_degree = np.array([3] * ndim)
        else:
            self.max_degree = np.array(max_degree)
        
        self.bounds = bounds
        self.ndim = ndim
        self.scaler = MinMaxScaler(self.ndim, self.bounds)
                
    def compute(self,dataset, policy, discount_idx = None, indexes = None):
        '''
        Computes the ortogonality matrix of dimensions
        (max_degree[0] + 1) x (max_degree[1] + 1) x ... x (max_degree[ndim-1] + 1)
        Each entry in position (i1,i2,...,i_ndim-1) is the normalized scalar product
        between the gradient of log policy and the Chebichev polynomial in ndim dimensions
        of degrees (i1,i2,...,i_ndim-1)
        '''
        
        if discount_idx is not None:
            raise Exception('Not implemented yet!')
            
        if indexes == None:
            indexes = np.arange(0,self.ndim)
            
        self.features = dataset[:,indexes]
        self.scaled_features = self.scaler.scale(self.features)
        
        #Gradients are computed over the dataset not scaled
        self.gradient = sample_log_gradients(policy,self.features)
        self.gradient_norms = sample_norms(self.gradient)
        
        #Chebichev polys ara evaluated over the scaled dataset
        self.poly = chebvanderNd(self.scaled_features,self.ndim,self.max_degree)
        self.poly_norms = sample_norms(self.poly)
        
        self.normalized_gradients = self.gradient/self.gradient_norms
        self.normalized_poly = self.poly/self.poly_norms
        
        self.ort_matrix = np.tensordot(self.normalized_gradients.T, self.normalized_poly, axes=1)
        return self.ort_matrix
    
    def get_ort_matrix(self):
        return self.ort_matrix
    
    
        
        

        