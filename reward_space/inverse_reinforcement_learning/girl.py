import numpy as np
from cvxopt import matrix, solvers
from reward_space.policy_gradient.gradient_estimator import MaximumLikelihoodEstimator

class LinearGIRL(object):

    '''
    Pirotta, Matteo, and Marcello Restelli.
    "Inverse Reinforcement Learning through Policy Gradient Minimization."
    AAAI. 2016.
    '''

    def __init__(self,
                 trajectories,
                 reward_features,
                 policy_gradient):

        self.trajectories = trajectories
        self.policy_gradient = policy_gradient
        self.reward_features = reward_features
        self.n_features = reward_features.shape[1]

    def fit(self):
        ml_estimator = MaximumLikelihoodEstimator(self.trajectories)
        gradient = ml_estimator.estimate_gradient(self.reward_features,
                                       self.policy_gradient,
                                       True).T

        print(gradient)

        P = matrix(np.dot(gradient.T, gradient))
        q = matrix(np.zeros((self.n_features, 1)))

        #Q = matrix(np.eye(self.n_features))
        #h = matrix(np.zeros(self.n_features, 1))
        A = matrix(np.ones((1, self.n_features)))
        b = matrix(-np.ones((1, 1)))

        res = solvers.qp(P, q, A=A, b=b)
        #res = solvers.qp(P, q, Q, h)
        return np.array(res['x']).ravel()