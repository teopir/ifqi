import numpy as np

def add_discount(dataset, horizon, discount_factor):
    new_dataset = np.zeros(shape=(dataset.shape[0], dataset.shape[1]+1))
    new_dataset[:,:-1] = dataset
    t = 0
    for i in range(new_dataset.shape[0]):
        new_dataset[i,5] = discount_factor**t
        t += 1
        if new_dataset[i,4] == 1 or t == horizon:
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
            
            
            
            
            
            