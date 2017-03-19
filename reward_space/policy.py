import numpy as np
from gym.utils import seeding
from taxi_policy_iteration import compute_policy

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
        #bound_action = self.check_action_bounds(action)
        #return bound_action
        return action
    
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
        policy = compute_policy()
        self.nS = 500
        self.nA = 6
        self.PI = np.zeros((self.nS, self.nA))
        self.PI2 = np.zeros((self.nS, self.nS * self.nA))
        s = np.array(policy.keys())
        a = np.array(policy.values())
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