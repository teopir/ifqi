from gym.utils import seeding
from scipy.stats import multivariate_normal
import numpy as np

class Policy(object):

    '''
    Abstract cass
    '''

    def draw_action(self, states, absorbing, evaluation=False):
        pass

    def pdf(self, states, actions):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

class ParametricPolicy(Policy):

    '''
    Abstract class for parametric policies
    '''

    def gradient(self, states, actions):
        pass

    def gradient_log(self):
        pass

    def hessian(self, states, actions):
        pass

    def hessian_log(self):
        pass

class GaussianPolicy(ParametricPolicy):

    '''
    Multivariate Gaussian policy
    a \sim N(\phi(s)^T \theta, \Sigma)
    '''

    def __init__(self,
                 n_dimensions,
                 n_parameters,
                 covariance_matrix,
                 parameters,
                 state_features):

        '''
        Constructor
        :param n_dimensions: number of dimensions
        :param n_parameters: number of parameters
        :param covariance_matrix: (n_dimensions,n_dimensions) symmetric positive
            semidefinite matrix
        :param parameters: (n_parameters, n_dimensions) matrix
        :param state_features: function that returns for each state (n_dimensions)
            vector a (n_paramaters) vector
        '''

        self.n_dimensions = n_dimensions
        self.n_parameters = n_parameters
        self.covariance_matrix = np.array(covariance_matrix, ndmin=2)
        self.parameters = np.array(parameters, ndmin=2)
        self.state_features = state_features

        self.inverse_covariance_matrix = np.linalg.inv(self.covariance_matrix)

        self.seed()

    def pdf(self, states, actions):
        return map(self._pdf, list(states), list(actions))

    def _pdf(self, state, action):
        state = np.array(state, ndmin=1)
        action = np.array(action, ndmin=1)
        mean = np.dot(self.parameters, self.state_features(state)[:, np.newaxis])
        return multivariate_normal.pdf(action, mean, self.covariance_matrix)

    def _gradient_log(self, state, action):
        state = np.array(state, ndmin=1)
        action = np.array(action, ndmin=1)
        feature = self.state_features(state)[:, np.newaxis]
        gradient = np.linalg.multi_dot([self.inverse_covariance_matrix, \
                    (action - np.dot(self.parameters, feature)), feature.T])
        return gradient

    def gradient_log(self, states, actions):
        return map(self._gradient_log, list(states), list(actions))

    def _hessian_log(self, state, action):
        NotImplementedError()

    def draw_action(self, states, absorbing, evaluation=False):
        state = np.array(states, ndmin=1)
        mean = np.dot(self.parameters, self.state_features(state)[:, np.newaxis]).ravel()
        action = self.np_random.multivariate_normal(mean, self.covariance_matrix)
        return action