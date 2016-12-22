"""Advertising MDP"""
from gym.utils import seeding
import numpy as np

from environment import Environment

class Bandit(Environment):

    R = np.array([2,10,3])
        
    def __init__(self, gamma=0.9):
        self.horizon = 100
        self.gamma = gamma
        # initialize state
        self.seed()
        self.reset()

    def step(self, action, render=False):
        action = np.asscalar(action)
        reward = self.R[action]
        return self.get_state(), reward, False, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None):
        self.state = np.array(0, ndmin=1)
        return self.get_state()

    def get_state(self):
        return self.state
    
    def computeQFunction(self, policy):
        return np.linalg.solve(np.eye(3) - self.gamma * np.dot(np.ones((3,1)), policy.get_distribution()), self.R)
