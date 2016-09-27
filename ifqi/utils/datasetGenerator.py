"""
This file create, manage, and save on disk a dataset
"""
import numpy as np
import ifqi.evaluation.evaluation as evaluate

class DatasetGenerator:

    def __init__(self, environment, policy, n_episodes):
        self.environment = environment
        self._stateDim = self.environment.observation_space.shape[0]
        self.data = np.zeros((0,3 + self._stateDim * 2 + 1))
        
        for _ in xrange(n_episodes):
            tempData = evaluate.collectEpisode(environment, policy)
            self.data = np.concatenate((self.data,tempData), axis=0)
        
    def save(self,fileName):
        np.save(fileName, self.data)
        
    def load(self,fileName):
        self.data = np.load(fileName)
        
    def loadAppend(self, fileName):
        tempData = np.load(fileName)
        self.data = np.concatenate((self.data, tempData))
        
    def generateAppend(self, policy=None, n_episodes=100):
        for _ in xrange(n_episodes):
            tempData = evaluate.collectEpisode(self.environment, policy)
            self.data = np.concatenate((self.data,tempData), axis=0)
            
    def generate(self, policy=None, n_episodes=100):
        self.reset()
        for _ in xrange(n_episodes):
            tempData = evaluate.collectEpisode(self.environment, policy)
            self.data = np.concatenate((self.data,tempData), axis=0)
        
            
    def reset(self):
        self.data = np.zeros((0,))
        
    @property
    def action(self):
        return self.data[:, 1+self._stateDim]
    
    @property
    def state(self):
        return self.data[:, 1:1 + self._stateDim]

    @property
    def nextState(self):
        return self.data[:, self._stateDim + 3: 2 * self._stateDim + 3]

    @property
    def absorbing(self):
        return self.data[:, -1]
    
    @property
    def endEpisode(self):
        return self.data[:,0]
        
    @property
    def reward(self):
        return self.data[:, self._stateDim + 2]
    
    @property
    def sar(self):
        return (np.concatenate((self.state,self.action), axis=1),self.reward)
    
    