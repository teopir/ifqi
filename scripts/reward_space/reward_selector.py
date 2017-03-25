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
         a tensor of n_sample x (deg[0] + 1) x ... x (deg[n_dim-1] + 1)
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
    samples = samples**2
    return np.sqrt(samples.sum(axis=0))

def sample_log_gradients(policy,samples):
    return np.array(map(lambda v: policy.gradient_log_pdf(v[0],v[1]), samples))


class RewardSelector(object):
    
    def __init__(self, ndim, bounds, max_degree=3):
        self.max_degree = max_degree
        self.bounds = bounds
        self.ndim = ndim
        self.scaler = MinMaxScaler(self.ndim, self.bounds)
                
    def compute(self,samples,policy):
        scaled_samples = self.scaler.scale(samples)
        
        sample_gradient = sample_log_gradients(policy,samples)
        sample_gradient_norms = sample_norms(sample_gradient)
        
        sample_poly = chebvanderNd(scaled_samples,self.ndim,[self.max_degree,self.max_degree])
        sample_poly_norms = sample_norms(sample_poly)
        
        normalized_sample_gradients = sample_gradient/sample_gradient_norms
        normalized_sample_poly = sample_poly/sample_poly_norms
        
        ort = np.tensordot(normalized_sample_gradients.T, normalized_sample_poly, axes=1)
        return ort
        

        