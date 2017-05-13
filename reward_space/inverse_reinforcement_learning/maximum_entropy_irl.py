import numpy.linalg as la
from reward_space.inverse_reinforcement_learning.utils import *


class MaximumEntropyIRL(object):

    '''
    Ziebart, Brian D., et al.
    "Maximum Entropy Inverse Reinforcement Learning."
    AAAI. Vol. 8. 2008.
    '''

    eps = 1e-24

    def __init__(self,
                 reward_features,
                 trajectories,
                 transition_model,
                 initial_distribution,
                 gamma,
                 horizon,
                 learning_rate=0.01,
                 max_iter=100,
                 type_='state',
                 gradient_method='linear',
                 evaluation_horizon=100):

        # transition model: tensor (n_states, n_actions, n_states)

        self.reward_features = reward_features
        self.trajectories = trajectories
        self.transition_model = transition_model
        self.initial_distribution = initial_distribution
        self.gamma = gamma
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        if not type_ in ['state', 'state-action']:
            raise ValueError()
        self.type_ = type_

        if not gradient_method in ['linear', 'exponentiated']:
            raise ValueError()
        self.gradient_method = gradient_method

        self.evaluation_horizon = evaluation_horizon

        self.n_states, self.n_actions = transition_model.shape[:2]
        self.n_states_actions = self.n_states * self.n_actions
        self.n_features = reward_features.shape[1]

    def compute_expected_state_visitation_frequency(self, horizon, reward):

        #Backward pass
        Z_s = np.ones(self.n_states)
        Z_sa = np.zeros((self.n_states, self.n_actions))

        #To avoid bad arithmetic approximations
        reward -= max(reward)
        reward_exp = np.exp(reward)
        reward_exp /= reward_exp.sum()

        for t in range(horizon):
            Z_sa = np.tensordot(self.transition_model, reward_exp * Z_s, axes=1)
            Z_s = Z_sa.sum(axis=1)

        #Local probability computation
        P = Z_sa / (self.eps + Z_s[:, np.newaxis])
        #Forward pass
        D_st = np.tile(self.initial_distribution[:, np.newaxis], horizon)
        for t in range(horizon - 1):
            D_st[:, t + 1] = (P * np.tensordot(self.transition_model, D_st[:, t], axes=1)).sum(axis=1)

        D_s = D_st.sum(axis=1).ravel()
        return D_s

    def compute_expected_state_action_visitation_frequency(self, horizon, reward):

        #Backward pass
        Z_s = np.ones(self.n_states)
        Z_sa = np.zeros((self.n_states, self.n_actions))

        #To avoid bad arithmetic approximations
        reward -= max(reward)
        reward_exp = np.exp(reward)
        reward_exp /= reward_exp.sum()
        reward_exp = reward_exp.reshape(Z_sa.shape)

        for t in range(horizon):
            Z_sa = (self.transition_model * (reward_exp.T * Z_s)).sum(axis=2)
            Z_s = Z_sa.sum(axis=1)

        #Local probability computation
        P = Z_sa / (self.eps + Z_s[:, np.newaxis])
        P2 = np.zeros((self.n_states, self.n_states_actions))
        P2[np.repeat(np.arange(self.n_states), self.n_actions), np.arange(
            self.n_states_actions)] = P.ravel()

        #Forward pass
        D_sa_t = np.tile(np.dot(self.initial_distribution, P2)[:, np.newaxis], horizon)
        D_st = np.tile(self.initial_distribution[:, np.newaxis], horizon)
        for t in range(horizon - 1):
            D_st[:, t + 1] = (P * np.tensordot(self.transition_model, D_st[:, t], axes=1)).sum(axis=1)
            D_sa_t[:, t + 1] = np.dot(D_st[:, t + 1], P2)

        D_sa = D_sa_t.sum(axis=1).ravel()
        return D_sa


    '''
    def compute_expected_state_visitation_frequency(self, horizon, reward):

        #Backward pass
        Z_s = np.ones(self.n_states)
        Z_sa = np.zeros((self.n_states, self.n_actions))
        for t in range(horizon):
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    Z_sa[s, a] = (self.transition_model[s, a] * np.exp(reward[s]) * Z_s).sum()
                Z_s[s] = Z_sa[s, :].sum()

        #Local probability computation
        P = Z_sa / Z_s[:, np.newaxis]

        #Forward pass
        D_st = np.tile(self.initial_distribution[:, np.newaxis], horizon)
        for t in range(horizon - 1):
            for s in range(self.n_states):
                D_st[s, t + 1] = la.multi_dot([P[s, :], self.transition_model[s, :, :], D_st[:, t].T])

        D_s = D_st.sum(axis=1).ravel()
        return D_s
    '''
    '''
    def compute_expected_state_action_visitation_frequency(self, horizon, reward):

        #Backward pass

        Z_s = np.ones(self.n_states)
        Z_sa = np.zeros((self.n_states, self.n_actions))
        reward_exp = np.exp(reward)
        reward_exp /= reward_exp.sum()
        print(reward_exp)
        for t in range(horizon):
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    Z_sa[s, a] = (self.transition_model[s, a] * reward_exp[s] * Z_s).sum()
                Z_s[s] = Z_sa[s, :].sum()

        #Local probability computation
        P = Z_sa / Z_s[:, np.newaxis]
        P2 = np.zeros((self.n_states, self.n_states_actions))
        P2[np.repeat(np.arange(self.n_states), self.n_actions), np.arange(self.n_states_actions)] = P.ravel()

        #Forward pass
        D_sa_t = np.tile(np.dot(self.initial_distribution, P2)[:, np.newaxis], horizon)
        D_st = np.tile(self.initial_distribution[:, np.newaxis], horizon)
        for t in range(horizon - 1):
            for s in range(self.n_states):
                index = s * self.n_actions + a
                D_st[s, t + 1] = la.multi_dot([P[s, :], self.transition_model[s, :, :], D_st[:, t].T])
                D_sa_t[index, t + 1] = D_st[s, t + 1] * P[s]

        D_s = D_st.sum(axis=1).ravel()
        D_sa_t = D_sa_t.sum(axis=1).ravel()
        return D_s
    '''

    def fit(self, verbose=False):

        #Compute features expectations
        feature_expectations = compute_feature_expectations(self.reward_features,
                                                            self.trajectories,
                                                            np.arange(self.n_states),
                                                            np.arange(self.n_actions),
                                                            self.gamma,
                                                            self.horizon,
                                                            self.type_)
        #Weights initialization
        w = np.ones(self.n_features) / self.n_features

        #Gradient descent
        for i in range(self.max_iter):
            if verbose:
                print('Iteration %s/%s' % (i + 1, self.max_iter))
            reward = np.dot(self.reward_features, w)

            if self.type_ == 'state':
                expected_vf = self.compute_expected_state_visitation_frequency(self.evaluation_horizon, reward)
            else:
                expected_vf = self.compute_expected_state_action_visitation_frequency(self.evaluation_horizon, reward)

            gradient = feature_expectations - np.dot(self.reward_features.T, expected_vf)

            if self.gradient_method == 'linear':
                w +=  self.learning_rate * gradient
            else:
                exponential = np.exp(self.learning_rate * gradient)
                w = w * exponential / np.dot(w, exponential).sum()

        w /= w.sum()
        reward = np.dot(self.reward_features, w)
        return reward.ravel()