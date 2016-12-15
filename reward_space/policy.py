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
        return np.array(np.array(state/np.power(self.sigma,2)*(action - np.dot(self.K,state))).ravel(), ndmin=1)
        
        
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
    
    
    
class AdvertisingPolicy(SimplePolicy):
    
    policy = np.array([[1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1]])
    
    def __init__(self):
        self.seed()
    
    def draw_action(self, state, done):
        action = self.np_random.choice(5, p=self.policy[np.asscalar(state)])
        return action
        
    def get_distribution(self):
        return self.policy
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)


def energy(state,action,K):
    return (state == 0)*K[0] + (state == 1)*K[1]
    
def epsilon_sigmoid(state, action, eps, K):
    return eps + (1-2*eps)/(1+np.exp(-energy(state,action,K)))

def gradient_epsilon_sigmoid(state, action, eps, K):
    _energy = energy(state,action,K)
    return (1-2*eps)*np.exp(-_energy)/((np.exp(-_energy)+1)**2*((1-2*eps)/(np.exp(-_energy)+1)+eps))

def gradient_1_epsilon_sigmoid(state, action, eps, K):
    _energy = energy(state,action,K)
    return -(1-2*eps)*np.exp(-_energy)/((np.exp(-_energy)+1)**2*(-(1-2*eps)/(np.exp(-_energy)+1)+1-eps))         
        
class AdvertisingSigmoidPolicy(SimplePolicy):
    
    def __init__(self, K, eps):
        self.seed()
        self.K = np.array(K, ndmin=1)
        self.eps = eps
        self._build_policy()
    
    def _build_policy(self):
        self.policy = np.array([[epsilon_sigmoid(0,0,self.eps,self.K), 1-epsilon_sigmoid(0,1,self.eps,self.K), 0, 0, 0],
                                [0, 0, epsilon_sigmoid(1,2,self.eps,self.K), 1-epsilon_sigmoid(1,3,self.eps,self.K), 0],
                                [0, 0, 0, 0, 1]])
        
    def draw_action(self, state, done):
        action = self.np_random.choice(5, p=self.policy[np.asscalar(state)])
        return action
        
    def get_distribution(self):
        return self.policy
        
    def gradient_log_pdf(self):
        return np.array([[gradient_epsilon_sigmoid(0, 0, self.eps, self.K), gradient_1_epsilon_sigmoid(0, 1, self.eps, self.K), 0, 0, 0],
                         [0, 0, gradient_epsilon_sigmoid(1, 2, self.eps, self.K), gradient_1_epsilon_sigmoid(1, 3, self.eps, self.K), 0]])
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

def bounded_gaussian(eps, theta):
    return eps + (1-2*eps) * np.exp(-np.power(theta,2)) 

def gradient_log_bounded_gaussian(eps, theta):
    return -2 * theta * (1-2*eps) * np.exp(-np.power(theta,2)) / bounded_gaussian(eps, theta)
        
class AdvertisingGaussianPolicy(SimplePolicy):
    
    def __init__(self, eps, theta1, theta2):
        self.seed()
        self.theta1 = theta1
        self.theta2 = theta2
        self.eps = eps
        self._build_policy()
    
    def _build_policy(self):
        self.policy = np.array([[bounded_gaussian(self.eps, self.theta1), 1-bounded_gaussian(self.eps, self.theta1), 0, 0, 0],
                                [0, 0, bounded_gaussian(self.eps, self.theta2), 1-bounded_gaussian(self.eps, self.theta2), 0],
                                [0, 0, 0, 0, 1]])
        
    def draw_action(self, state, done):
        action = self.np_random.choice(5, p=self.policy[np.asscalar(state)])
        return action
        
    def get_distribution(self):
        return self.policy
        
    def gradient_log_pdf(self):
        return np.array([[gradient_log_bounded_gaussian(self.eps, self.theta1), -gradient_log_bounded_gaussian(self.eps, self.theta1), 0, 0, 0],
                         [0, 0, gradient_log_bounded_gaussian(self.eps, self.theta2), -gradient_log_bounded_gaussian(self.eps, self.theta2), 0]])
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)