import numpy as np


class MountainCarPreprocessor(object):    
        
    def preprocess(self, state):
        actions = (state[:, 2]).reshape(-1, 1)
        return np.concatenate((state[:, 0:2], actions), axis=1)
