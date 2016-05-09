import numpy as np


class InvertedPendulumPreprocessor(object):    
        
    def preprocess(self, state):
        actions = (state[:,2]-1).reshape(-1,1)
        return np.concatenate((state[:,0:2],actions),axis=1)
