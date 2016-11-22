import numpy as np
from gym.utils import seeding

class Policy(object):
    
    '''
    Abstract class
    '''

    def draw_action(self, state, done):
        pass

class SimplePolicy(Policy):
    
    '''
    Deterministic policy with parameter K
    '''
    
    def __init__(self, K, action_bounds=None):
        self.K = np.array(K, ndmin=2)
        self.n_dim = self.K.shape[0]
        if action_bounds is None:
            self.action_bounds = np.array([[-np.inf] *self.n_dim, [np.inf] *self.n_dim], ndmin=2)
        else:
            self.action_bounds = np.array(action_bounds)
    
    def draw_action(self, state, done):
        action = np.dot(self.K, state)
        bound_action = self.check_action_bounds(action)
        return bound_action
    
    def check_action_bounds(self, action):
        return np.clip(action, self.action_bounds[0], self.action_bounds[1])

        
class GaussianPolicy1D(SimplePolicy):
    
    '''
    Gaussian policy with parameter K for the mean and fixed variance
    just for 1 dimension lqr
    '''

    def __init__(self, K, sigma, action_bounds=None):
        SimplePolicy.__init__(self, K, action_bounds)
        self.sigma = sigma
        self.seed()

    def draw_action(self, state, done):
        state = np.array(state, ndmin=1)
        action = np.dot(self.K, state) + self.np_random.randn() * self.sigma
        bound_action = self.check_action_bounds(action)
        return bound_action
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        
    def gradient_log_pdf(self, state, action):
        return np.array(np.array(state/self.sigma*(action - np.dot(self.K,state))).ravel(), ndmin=1)
        
        
class GaussianPolicy(SimplePolicy):
    
    '''
    Gaussian policy with parameter K for the mean and fixed variance
    for any dimension
    TBR
    '''
    
    def __init__(self, K, covar, action_bounds=None):
        SimplePolicy.__init__(self, K, action_bounds)
        self.covar = np.array(covar, ndmin=2)
        self.seed()

    def draw_action(self, state, done):
        state = np.array(state, ndmin=1)
        mean = np.dot(self.K, state)
        action = self.np_random.multivariate_normal(mean, self.covar)
        bound_action = self.check_action_bounds(action)
        return bound_action
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
    '''
    TODO: check the following code
    def gradient_log_pdf(self,state,action):
        state = np.array(state, ndmin=1)
        action = np.array(action, ndmin=1)
        return np.array(state.dot(action - np.dot(self.K,state)).dot(np.linalg.inv(self.cov)), ndmin=1)
    '''