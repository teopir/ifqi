from reward_space.inverse_reinforcement_learning.utils import *
from cvxopt import matrix, solvers

class LPAL(object):

    '''
    LPAL (Linear Programming Apprenticeship Learning)
    Syed, Umar, Michael Bowling, and Robert E. Schapire.
    "Apprenticeship learning using linear programming."
    Proceedings of the 25th international conference on Machine learning. ACM, 2008.
    '''

    def __init__(self,
                 reward_features,
                 trajectories,
                 transition_model,
                 initial_distribution,
                 gamma,
                 horizon):
        self.reward_features = reward_features
        self.trajectories = trajectories
        self.initial_distribution = initial_distribution
        self.transition_model = transition_model
        self.gamma = gamma
        self.horizon = horizon
        self.n_samples = trajectories.shape[0]
        self.n_trajectories = int(trajectories[:, -1].sum())
        self.n_states = transition_model.shape[1]
        self.n_states_actions = reward_features.shape[0]
        self.n_features = reward_features.shape[1]
        self.n_actions = self.n_states_actions / self.n_states


    def fit(self, verbose=False):

        #Computing epsilon-good estimate of value function (feature expectations)
        feature_expectations = compute_feature_expectations(self.reward_features,
                                                            self.trajectories,
                                                            np.arange(self.n_states),
                                                            np.arange(self.n_actions),
                                                            self.gamma,
                                                            self.horizon)

        #Build LP data structures
        c = matrix(np.concatenate([[-1], np.zeros(self.n_states_actions)]))

        #Inequality constraint
        G_in = np.hstack([np.ones((self.n_features, 1)), -self.reward_features.T])
        h_in = -feature_expectations

        #Non negativity constraint
        G_nn = np.hstack([np.zeros((self.n_states_actions, 1)), -np.eye(self.n_states_actions)])
        h_nn = np.zeros(self.n_states_actions)

        G = matrix(np.vstack([G_in, G_nn]))
        h = matrix(np.concatenate([h_in, h_nn]))

        #Equality constraint
        A_partial = np.repeat(np.eye(self.n_states), self.n_actions, axis=1) - \
                    self.gamma * self.transition_model.T
        A = matrix(np.hstack([np.zeros((self.n_states, 1)), A_partial]))
        b = matrix(self.initial_distribution)

        #Solve the LP problem
        solvers.options['show_progress'] = False
        result = solvers.lp(c, G, h, A, b)

        #Get the weights
        x = np.array(result['x'][1:]).ravel()

        #Output policy
        pi = x.reshape(self.n_states, self.n_actions)
        pi[pi < 1e-24] = 1e-24
        pi /= pi.sum(axis=1)[:, np.newaxis]

        return pi


