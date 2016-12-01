import numpy as np
from numpy.polynomial.chebyshev import chebvander, chebval

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
        prod *= chebval(x[i],c)
        c[d] = 0
    return prod

def add_discount(dataset, endofepisode_idx, discount_factor):
    new_dataset = np.zeros(shape=(dataset.shape[0], dataset.shape[1]+1))
    new_dataset[:,:-1] = dataset
    t = 0
    for i in range(new_dataset.shape[0]):
        new_dataset[i,-1] = discount_factor**t
        t += 1
        if new_dataset[i,endofepisode_idx] == 1:
            t = 0
    return new_dataset


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
            
            
            
            
            
            