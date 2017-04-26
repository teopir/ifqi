import numpy as np
from gym.utils import seeding
from taxi_policy_iteration import compute_policy
from scipy.stats import multivariate_normal
import numpy.linalg as la

class Policy(object):
    
    '''
    Abstract class
    '''

    def draw_action(self, state, done):
        pass


class DiscreteGaussianPolicy(Policy):

    def __init__(self, ndim, mu, sigma, features, state_space, action_space):
        self.ndim = ndim
        self.mu = np.array(mu, ndmin=1)
        self.sigma = np.array(sigma, ndmin=2)
        self.features = features
        self.state_space = state_space
        self.action_space = action_space
        self.n_states = len(state_space)
        self.n_actions = len(action_space)

    def get_idxs(self, state, action):
        s_idx = np.argwhere(self.state_space == state)
        a_idx = np.argwhere(self.action_space == action)
        idx = s_idx * self.n_actions + a_idx
        idxs = s_idx + np.arange(self.n_actions)
        return idx, idxs

    def pdf(self, state, action):
        idx, idxs = self.get_idxs(state, action)
        num = multivariate_normal(self.features[idx], self.mu, self.sigma)
        den = np.sum(multivariate_normal(self.features[idxs], self.mu, self.sigma))
        return num / den

    def gradient_log(self, state, action):
        idx, idxs = self.get_idxs(state, action)
        num = np.sum((self.features[idx] - self.features[idxs]) * multivariate_normal(self.features[idxs], self.mu, self.sigma))
        den = np.sum(multivariate_normal(self.features[idxs], self.mu, self.sigma))
        return num / den

    def gradient(self, state, action):
        return self.gradient_log(state, action) * self.pdf(state, action)

    def hessian(self, state, action):
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

    def get_dim(self):
        return 1

    def set_parameter(self, K):
        self.K = K

    def draw_action(self, state, done):
        state = np.array(state, ndmin=1)
        action = np.dot(self.K, state) + self.np_random.randn() * self.sigma
        #bound_action = self.check_action_bounds(action)
        #return bound_action
        return action
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def pdf(self, state, action):
        return np.array(1. / np.sqrt(2 * np.pi * self.sigma ** 2) * np.exp(- 1. / 2 * (action - self.K * state) ** 2 / self.sigma ** 2), ndmin=1)


    def gradient_log(self, state, action, type_='state-action'):
        if type_ == 'state-action':
            return np.array(np.array(state/np.power(self.sigma,2)*(action - np.dot(self.K,state))).ravel(), ndmin=1)
        elif type_ == 'list':
            return map(lambda s,a: self.gradient_log(s, a), state, action)

    def hessian_log(self, state, action):
        return np.array(- state ** 2 / self.sigma ** 2, ndmin=2)
        
        
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

def scaled_gaussian(mu, sigma, min_, max_, i):
    return np.exp((-0.5*((i-mu)/sigma)**2)) / sum(np.exp((-0.5*((np.arange(min_, max_+1)-mu)/sigma)**2)))

def gradient_scaled_gaussian(mu, sigma, min_, max_, i):
    return i-mu - sum(((np.arange(min_, max_+1)-mu)/sigma) * np.exp((-0.5*((np.arange(min_, max_+1)-mu)/sigma)**2)))/sum(np.exp(((-0.5*(np.arange(min_, max_+1)-mu)/sigma)**2)))

class BanditPolicy(SimplePolicy):
    def __init__(self, mu):
        self.seed()
        self.mu = mu
        self._build_policy()

    def _build_policy(self):
        self.policy = np.array([[scaled_gaussian(self.mu,1.,1,3,1),
                                scaled_gaussian(self.mu,1.,1,3,2),
                                scaled_gaussian(self.mu,1.,1,3,3)]], ndmin=2)

    def draw_action(self, state, done):
        action = self.np_random.choice(3, p=self.policy[np.asscalar(state)])
        return action

    def get_distribution(self):
        return self.policy

    def gradient_log_pdf(self):
        return np.array([[gradient_scaled_gaussian(self.mu,1.,1,3,1),
                         gradient_scaled_gaussian(self.mu,1.,1,3,2),
                         gradient_scaled_gaussian(self.mu,1.,1,3,3)]], ndmin=2)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)


class TaxiEnvPolicy(SimplePolicy):

    def __init__(self):
        self.policy = compute_policy()
        self.nS = 500
        self.nA = 6
        self.PI = np.zeros((self.nS, self.nA))
        self.PI2 = np.zeros((self.nS, self.nS * self.nA))
        s = np.array(self.policy.keys())
        a = np.array(self.policy.values())
        self.PI[s, a] = 1.
        self.PI2[s, s * self.nA + a] = 1.
        self.seed()

    def draw_action(self, state, done):
        action = self.np_random.choice(6, p=self.PI[np.asscalar(state)])
        return action

    def get_distribution(self):
        return self.PI2

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

class TaxiEnvRandomPolicy(SimplePolicy):

    def __init__(self):
        self.nS = 500
        self.nA = 6
        self.PI = np.ones((self.nS, self.nA)) / self.nA
        self.PI2 = np.zeros((self.nS, self.nS * self.nA))
        self.PI2[np.repeat(np.arange(self.nS),self.nA), np.arange(self.nS*self.nA)] = 1. / self.nA
        self.seed()

    def draw_action(self, state, done):
        action = self.np_random.choice(6, p=self.PI[np.asscalar(state)])
        return action

    def get_distribution(self):
        return self.PI2

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)


class TaxiEnvPolicyOneParameter(SimplePolicy):

    def __init__(self, mu, sigma, opt_idx):
        self.opt_policy = compute_policy()
        self.opt_idx = opt_idx
        self.mu = mu
        self.sigma = sigma
        self.nS = 500
        self.nA = 6
        self.PI = np.zeros((self.nS, self.nA))
        self.PI2 = np.zeros((self.nS, self.nS * self.nA))
        self.C = np.zeros((self.nS * self.nA, 1))

        idx1 = np.delete(np.arange(1,self.nA+1), opt_idx-1)
        for i in range(self.nS):
            opt_a= self.opt_policy[i]
            self.PI[i,opt_a] = scaled_gaussian(self.mu,self.sigma, 1,self.nA,opt_idx)
            self.PI2[i,i*self.nA + opt_a] = self.PI[i,opt_a]
            self.C[i*self.nA + opt_a, 0] = gradient_scaled_gaussian(self.mu, self.sigma, 1, self.nA, opt_idx)
            idx2 = np.delete(np.arange(self.nA), opt_a)
            for j,jidx in zip(idx2,idx1):
                self.PI[i, j] = scaled_gaussian(self.mu, self.sigma, 1, self.nA, jidx)
                self.PI2[i, i * self.nA + j] = self.PI[i, j]
                self.C[i * self.nA + j, 0] = gradient_scaled_gaussian(self.mu, self.sigma, 1, self.nA, jidx)
        self.seed()

    def get_distribution(self):
        return self.PI2

    def draw_action(self, state, done):
        action = self.np_random.choice(6, p=self.PI[np.asscalar(state)])
        return action

    def gradient_log_pdf(self):
        return self.C

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

class TaxiEnvPolicyStateParameter(SimplePolicy):

    def __init__(self, mu, sigma, opt_idx):
        self.opt_policy = compute_policy()
        self.opt_idx = opt_idx
        self.mu = mu
        self.sigma = sigma
        self.nS = 500
        self.nA = 6
        self.PI = np.zeros((self.nS, self.nA))
        self.PI2 = np.zeros((self.nS, self.nS * self.nA))
        self.C = np.zeros((self.nS * self.nA, self.nS))

        idx1 = np.delete(np.arange(1,self.nA+1), opt_idx-1)
        for i in range(self.nS):
            opt_a= self.opt_policy[i]
            self.PI[i,opt_a] = scaled_gaussian(self.mu,self.sigma, 1,self.nA,opt_idx)
            self.PI2[i,i*self.nA + opt_a] = self.PI[i,opt_a]
            self.C[i*self.nA + opt_a, i] = gradient_scaled_gaussian(self.mu, self.sigma, 1, self.nA, opt_idx)
            idx2 = np.delete(np.arange(self.nA), opt_a)
            for j,jidx in zip(idx2,idx1):
                self.PI[i, j] = scaled_gaussian(self.mu, self.sigma, 1, self.nA, jidx)
                self.PI2[i, i * self.nA + j] = self.PI[i, j]
                self.C[i * self.nA + j, i] = gradient_scaled_gaussian(self.mu, self.sigma, 1, self.nA, jidx)
        self.seed()

    def get_distribution(self):
        return self.PI2

    def draw_action(self, state, done):
        action = self.np_random.choice(6, p=self.PI[np.asscalar(state)])
        return action

    def gradient_log_pdf(self):
        return self.C

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
'''
class TaxiEnvPolicy2StateParameter(SimplePolicy):

    def __init__(self, sigma):
        self.opt_policy = compute_policy()
        self.nS = 500
        self.nA = 6
        self.PI = np.zeros((self.nS, self.nA))
        self.PI2 = np.zeros((self.nS, self.nS * self.nA))
        self.C = np.zeros((self.nS * self.nA, 2*self.nS))

        for i in range(self.nS):
            opt_a= self.opt_policy[i]
            self.PI[i,opt_a] = 1. / (1. + (self.nA - 1) * np.exp(-0.5/sigma))
            self.PI2[i,i*self.nA + opt_a] = self.PI[i,opt_a]
            self.C[i*self.nA + opt_a, 2*i] = 0.
            self.C[i*self.nA + opt_a, 2*i+1] = 0.
            idx2 = np.delete(np.arange(self.nA), opt_a)
            for ind,j in enumerate(idx2):
                self.PI[i, j] = np.exp(-0.5/sigma) / (1. + (self.nA - 1) * np.exp(-0.5/sigma))
                self.PI2[i, i * self.nA + j] = self.PI[i, j]
                self.C[i * self.nA + j, 2*i] = np.cos(ind * 2 * np.pi / (self.nA - 1))/sigma
                self.C[i * self.nA + j, 2*i+1] = np.sin(ind * 2 * np.pi / (self.nA - 1))/sigma
        self.seed()

    def get_distribution(self):
        return self.PI2

    def draw_action(self, state, done):
        action = self.np_random.choice(6, p=self.PI[np.asscalar(state)])
        return action

    def gradient_log_pdf(self):
        return self.C

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
'''
class TaxiEnvPolicy2StateParameter(SimplePolicy):

    def __init__(self, sigma):
        self.opt_policy = compute_policy()
        self.nS = 500
        self.nA = 6
        self.PI = np.zeros((self.nS, self.nA))
        self.PI2 = np.zeros((self.nS, self.nS * self.nA))
        self.C = np.zeros((self.nS * self.nA, 2 * self.nS))
        self.H = np.zeros((2 * self.nS, 2 * self.nS, self.nS * self.nA))

        for i in range(self.nS):
            opt_a = self.opt_policy[i]
            self.PI[i, opt_a] = 1. / (1. + (self.nA - 1) * np.exp(-0.5 / sigma))
            self.PI2[i, i * self.nA + opt_a] = self.PI[i, opt_a]
            self.C[i * self.nA + opt_a, 2 * i] = 0.
            self.C[i * self.nA + opt_a, 2 * i + 1] = 0.

            h_cos = -sum(np.cos(2 * np.pi * np.arange(0, self.nA-1) / (self.nA - 1)) ** 2) * np.exp(-0.5 / sigma) / (1. + (self.nA - 1) * np.exp(-0.5 / sigma))
            h_sin = -sum(np.sin(2 * np.pi * np.arange(0, self.nA-1) / (self.nA - 1)) ** 2) * np.exp(-0.5 / sigma) / (1. + (self.nA - 1) * np.exp(-0.5 / sigma))
            h_cos_sin = -sum(np.cos(2 * np.pi * np.arange(0, self.nA-1) / (self.nA - 1)) * np.sin(2 * np.pi * np.arange(0, self.nA-1) / (self.nA - 1))) *np.exp(-0.5 / sigma) / (1. + (self.nA - 1) * np.exp(-0.5 / sigma))

            self.H[2 * i, 2 * i, i * self.nA + opt_a] = h_cos
            self.H[2 * i + 1, 2 * i + 1, i * self.nA + opt_a] = h_sin
            self.H[2 * i + 1, 2 * i, i * self.nA + opt_a] = h_cos_sin
            self.H[2 * i, 2 * i + 1, i * self.nA + opt_a] = h_cos_sin


            idx2 = np.delete(np.arange(self.nA), opt_a)
            for ind, j in enumerate(idx2):
                self.PI[i, j] = np.exp(-0.5 / sigma) / (1. + (self.nA - 1) * np.exp(-0.5 / sigma))
                self.PI2[i, i * self.nA + j] = self.PI[i, j]
                self.C[i * self.nA + j, 2 * i] = np.cos(ind * 2 * np.pi / (self.nA - 1)) / sigma
                self.C[i * self.nA + j, 2 * i + 1] = np.sin(ind * 2 * np.pi / (self.nA - 1)) / sigma

                self.H[2 * i, 2 * i, i * self.nA + j] = h_cos
                self.H[2 * i + 1, 2 * i + 1, i * self.nA + j] = h_sin
                self.H[2 * i + 1, 2 * i, i * self.nA + j] = h_cos_sin
                self.H[2 * i, 2 * i + 1, i * self.nA + j] = h_cos_sin

        self.seed()

    def get_distribution(self):
        return self.PI2

    def draw_action(self, state, done):
        action = self.np_random.choice(6, p=self.PI[np.asscalar(state)])
        return action

    def gradient_log_pdf(self):
        return self.C

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

class TaxiEnvPolicy2Parameter(SimplePolicy):

    def __init__(self, sigma):
        self.opt_policy = compute_policy()
        self.nS = 500
        self.nA = 6
        self.PI = np.zeros((self.nS, self.nA))
        self.PI2 = np.zeros((self.nS, self.nS * self.nA))
        self.C = np.zeros((self.nS * self.nA, 2))

        self.H = np.zeros((2, 2, self.nS * self.nA))
        h_cos = -sum(np.cos(2 * np.pi * np.arange(0, self.nA - 1) / (self.nA - 1)) ** 2) * np.exp(-0.5 / sigma) / (1. + (self.nA - 1) * np.exp(-0.5 / sigma))
        h_sin = -sum(np.sin(2 * np.pi * np.arange(0, self.nA - 1) / (self.nA - 1)) ** 2) * np.exp(-0.5 / sigma) / (1. + (self.nA - 1) * np.exp(-0.5 / sigma))
        h_cos_sin = -sum(np.cos(2 * np.pi * np.arange(0, self.nA - 1) / (self.nA - 1)) * np.sin(2 * np.pi * np.arange(0, self.nA - 1) / (self.nA - 1))) * np.exp(-0.5 / sigma) / (1. + (self.nA - 1) * np.exp(-0.5 / sigma))


        for i in range(self.nS):
            opt_a= self.opt_policy[i]
            self.PI[i,opt_a] = 1. / (1. + (self.nA - 1) * np.exp(-0.5/sigma))
            self.PI2[i,i*self.nA + opt_a] = self.PI[i,opt_a]
            self.C[i*self.nA + opt_a, 0] = 0.
            self.C[i*self.nA + opt_a, 1] = 0.

            self.H[0, 0, i * self.nA + opt_a] = h_cos
            self.H[1, 1, i * self.nA + opt_a] = h_sin
            self.H[0, 1, i * self.nA + opt_a] = h_cos_sin
            self.H[1, 0, i * self.nA + opt_a] = h_cos_sin

            idx2 = np.delete(np.arange(self.nA), opt_a)
            for ind,j in enumerate(idx2):
                self.PI[i, j] = np.exp(-0.5/sigma) / (1. + (self.nA - 1) * np.exp(-0.5/sigma))
                self.PI2[i, i * self.nA + j] = self.PI[i, j]
                self.C[i * self.nA + j, 0] = np.cos(ind * 2 * np.pi / (self.nA - 1))/sigma
                self.C[i * self.nA + j, 1] = np.sin(ind * 2 * np.pi / (self.nA - 1))/sigma

                self.H[0, 0, i * self.nA + j] = h_cos
                self.H[1, 1, i * self.nA + j] = h_sin
                self.H[0, 1, i * self.nA + j] = h_cos_sin
                self.H[1, 0, i * self.nA + j] = h_cos_sin

        self.seed()

    def get_pi(self):
        return self.PI2

    def draw_action(self, state, done):
        action = self.np_random.choice(6, p=self.PI[np.asscalar(state)])
        return action

    def gradient_log(self):
        return self.C

    def hessian_log(self):
        return self.H.T

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)


'''
TODO check the following code

from discrete_gaussian import DiscreteGaussian
class TaxiEnvPolicyStateParameter2(SimplePolicy):

    def __init__(self, ndim, sigma, theta=None):
        opt_policy = compute_policy()
        self.nS = 500
        self.nA = 6
        self.ndim = ndim
        self.sigma = sigma

        self.phi = np.zeros((self.nS, self.nA, self.ndim))
        opt_phi, phi_list = self._get_phi()

        self.PI = np.zeros((self.nS, self.nA))
        self.PI2 = np.zeros((self.nS, self.nS * self.nA))
        self.C = np.zeros((self.nS * self.nA, self.ndim*self.nS))

        for i in range(self.nS):
            opt_idx= opt_policy[i]
            other_idx = np.delete(np.arange(self.nA), opt_idx)
            self.phi[i,opt_idx] = opt_phi
            self.phi[i,other_idx] = phi_list


        if theta is None:
            self.theta = self._estimate_optimal_theta()
        else:
            self.theta = theta

        for i in range(self.nS):
            dg = DiscreteGaussian(self.phi[i, :], self.theta[i, :], self.sigma)
            for j in range(self.nA):
                self.PI[i, j] = dg.pdf(self.phi[i, j])
                self.PI2[i, i*self.nA + j] = self.PI[i, j]
                self.C[i*self.nA + j, self.ndim*i:self.ndim*(i+1)] = dg.grad_log(self.phi[i, j])

        self.seed()

    def _get_phi(self):
        return np.zeros((1,1,self.ndim)), np.zeors((1,self.nA - 1,self.ndim))

    def get_distribution(self):
        return self.PI2

    def draw_action(self, state, done):
        action = self.np_random.choice(6, p=self.PI[np.asscalar(state)])
        return action

    def gradient_log_pdf(self):
        return self.C

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
'''

class BoltzmannPolicy(Policy):

    def __init__(self, features, parameters):
        self.features = features
        self.parameters = parameters
        self.n_states = self.features.shape[0]
        self.n_actions = self.parameters.shape[0]
        self.n_parameters = self.features.shape[1]

        self.state_action_features = np.zeros((self.n_states * self.n_actions, self.n_actions * self.n_parameters))
        self.state_action_parameters = parameters.ravel()[:, np.newaxis]
        row_index = np.repeat(np.arange(self.n_states * self.n_actions), self.n_parameters)
        col_index = np.tile(np.arange(self.n_parameters * self.n_actions), self.n_states)
        features_repeated = np.repeat(features, self.n_actions, axis=0).ravel()
        self.state_action_features[row_index, col_index] = features_repeated

        self._build_density()
        self._build_grad_hess()

        self.seed()

    def set_parameter(self, new_parameter, build_gradient_hessian=True):
        self.state_action_parameters = np.copy(new_parameter)
        self.parameters = self.state_action_parameters.reshape((self.n_actions, self.n_parameters))

        self._build_density()
        if build_gradient_hessian:
            self._build_grad_hess()

        self.seed()

    def _build_density(self):
        numerators = np.exp(np.dot(self.features, self.parameters.T))
        denominators = np.sum(numerators, axis=1)[:, np.newaxis]

        self.pi = numerators / denominators
        self.pi2 = np.zeros((self.n_states, self.n_actions * self.n_states))
        row_index = np.arange(self.n_states)[:, np.newaxis]
        col_index = np.arange(self.n_states * self.n_actions).reshape(self.n_states, self.n_actions)
        self.pi2[row_index, col_index] = self.pi

    def _build_grad_hess(self):
        self.grad_log = np.zeros((self.n_states * self.n_actions,
                                  self.n_parameters * self.n_actions))
        self.hess_log = np.zeros((self.n_states * self.n_actions,
                                  self.n_parameters * self.n_actions,
                                  self.n_parameters * self.n_actions))
        #Compute the gradient for all (s,a) pairs
        for state in range(self.n_states):

            num = den = 0
            for action in range(self.n_actions):
                index = state * self.n_actions + action
                exponential = np.exp(np.dot(self.state_action_features[index],\
                                            self.state_action_parameters))
                num += self.state_action_features[index] * exponential
                den += exponential

            for action in range(self.n_actions):
                index = state * self.n_actions + action
                self.grad_log[index] = self.state_action_features[index] - num / den

        #Compute the hessian for all (s,a) pairs
        for state in range(self.n_states):

            num1 = num2 = den1 = 0
            for action in range(self.n_actions):
                index = state * self.n_actions + action
                exponential = np.exp(np.dot(self.state_action_features[index], \
                                            self.state_action_parameters))
                num1 += np.outer(self.state_action_features[index],\
                                 self.state_action_features[index]) * exponential
                num2 += self.state_action_features[index] * exponential
                den1 += exponential

            num = num1 * den1 - np.outer(num2, num2)
            den = den1 ** 2

            for action in range(self.n_actions):
                index = state * self.n_actions + action
                self.hess_log[index] = - num / den

    def pdf(self, state, action):
        num = np.exp(np.dot(self.features[state], self.parameters[action].T))
        den = np.sum(np.exp(np.dot(self.features[state], self.parameters.T)))
        return num / den

    def get_pi(self, type_='state-action'):
        if type_ == 'state-action':
            return self.pi2
        elif type_ == 'state':
            return self.pi
        elif type_ == 'function':
            return self.pdf
        else:
            raise NotImplementedError

    def gradient_log(self, states=None, actions=None, type_='state-action'):
        if type_ == 'state-action':
            return self.grad_log
        elif type_ == 'list':
            return np.array(map(lambda s,a: self.grad_log[int(s) * self.n_actions + int(a)], states, actions))
        else:
            raise NotImplementedError

    def hessian_log(self, type_='state-action'):
        if type_ == 'state-action':
            return self.hess_log
        else:
            raise NotImplementedError

    def draw_action(self, state, done):
        action = self.np_random.choice(self.n_actions, p=self.pi[np.asscalar(state)])
        return action

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def get_dim(self):
        return self.state_action_parameters.shape[0]

class TabularPolicy(Policy):

    def __init__(self, probability_table):
        self.pi = probability_table

        self.n_states, self.n_actions = probability_table.shape
        self.n_state_actions = self.n_actions * self.n_states
        self.pi2 = np.zeros((self.n_states, self.n_state_actions))

        rows = np.repeat(np.arange(self.n_states), self.n_actions)
        cols = np.arange(self.n_state_actions)
        self.pi2[rows, cols] = self.pi.ravel()

    def draw_action(self, state, done):
        action = self.np_random.choice(self.n_actions, p=self.pi[np.asscalar(state)])
        return action

    def get_distribution(self):
        return self.pi2

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)


