import numpy as np


class Identity(object):
    def fit_transform(self, phi_a, discrete_actions):
        basis = phi_a

        return basis

    def transform(self, phi_a, discrete_actions):
        basis = phi_a

        return basis


class AndCondition(object):
    def fit_transform(self, phi_a, discrete_actions):
        self.discrete_actions = discrete_actions

        # TODO: currently this works only with action_dim = 1
        state_size = phi_a.shape[1] - 1
        basis = np.zeros((phi_a.shape[0], state_size * self.discrete_actions.size))

        for i in range(self.discrete_actions.size):
            action = self.discrete_actions[i]
            idxs = np.all(phi_a[:, -1:] == action, axis=1)

            start = i * state_size
            basis[idxs, start:start + state_size] = phi_a[idxs, :-1]

        return basis

    def transform(self, phi_a):
        # TODO: currently this works only with action_dim = 1
        state_size = phi_a.shape[1] - 1
        basis = np.zeros((phi_a.shape[0], state_size * self.discrete_actions.size))

        for i in range(self.discrete_actions.size):
            action = self.discrete_actions[i]
            idxs = np.all(phi_a[:, -1:] == action, axis=1)

            start = i * state_size
            basis[idxs, start:start + state_size] = phi_a[idxs, :-1]

        return basis