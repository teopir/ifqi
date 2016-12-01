import numpy as np

from utils import MinMaxScaler, chebvanderNd  

def sample_norms(samples):
    square_samples = samples**2
    return np.sqrt(square_samples.sum(axis=0))

def sample_log_gradients(policy,features):
    return np.array(map(lambda v: policy.gradient_log_pdf(v[0],v[1]), features), ndmin=2)
       
class SampleGramSchmidt(object):
    
    def __init__(self, samples):
        self.samples = samples
    
    def compute(self, matrix, normalize=True):
        for i in range(matrix.shape[1]):
            v = matrix[:,i]
            for j in range(i):
                v -= self.project(v,matrix[:,j])
        if normalize:
           for i in range(matrix.shape[1]):
               matrix[:,i] = self.normalize(matrix[:,i])
        return matrix
    
    def project(self,v, u):
        return self.dot(v,u)/self.norm(u) * u
    
    def dot(self, v,u):
        res = 0
        for s in self.samples:
            res += v[s]*u[s]
        return res
    
    def norm(self, v):
        return self.dot(v,v)
        
    def normalize(self, v):
        return v/self.norm(v)
    
class OrthogonalPolynomialSelector(object):
    
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
            
        features = dataset[:,indexes]
        scaled_features = self.scaler.scale(features)
        
        #Gradients are computed over the dataset not scaled
        gradient = sample_log_gradients(policy,features)
        gradient_norms = sample_norms(gradient)
        
        #Chebichev polys ara evaluated over the scaled dataset
        poly = chebvanderNd(scaled_features,self.ndim,self.max_degree)
        poly_norms = sample_norms(poly)
        
        normalized_gradients = gradient/gradient_norms
        normalized_poly = poly/poly_norms
        
        self.ort_matrix = np.tensordot(normalized_gradients.T, normalized_poly, axes=1)
        return self.ort_matrix
    
    def get_ort_matrix(self):
        return self.ort_matrix
    
    
        
        

        