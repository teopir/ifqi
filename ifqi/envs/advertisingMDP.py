"""Advertising MDP"""
from gym.utils import seeding
import numpy as np

from environment import Environment

class AdvertisingMDP(Environment):
    
    #Action, new state revard R(a,s)
    R_sas = np.array([[0, 20, 0],
                      [-2, -27, 0],
                      [0, 20, 0],
                      [0, -5, -100],
                      [0, 0, 50]])
    
    P_sas = np.array([[0.9, 0.1, 0],
                      [0.3, 0.7, 0],
                      [0.4, 0.6, 0],
                      [0, 0.3, 0.7],
                      [0.2, 0, 0.8]])
    
    #Immediate expected reward
    R_sa = (R_sas*P_sas).sum(axis=1)
        
    def __init__(self, gamma=0.9):
        self.horizon = 100
        self.gamma = gamma
        # initialize state
        self.seed()
        self.reset()

    def step(self, action, render=False):
        action = np.asscalar(np.array(action))
        self.state = np.array([self.np_random.choice(3, p = self.P_sas[action])])
        reward = self.R_sas[action,self.state]
        return self.get_state(), reward, False, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None):
        if state is None:
            self.state = np.array([self.np_random.choice(3)])
        else:
            self.state = np.array([state]).ravel()

        return self.get_state()

    def get_state(self):
        return self.state
    
    def computeQFunction(self, policy):
        return np.linalg.solve(np.eye(5) - self.gamma * np.dot(self.P_sas, policy.get_distribution()), self.R_sa)
