from __future__ import print_function
import pandas as pd
import numpy as np


CONVERTERS = {"scalar": lambda x: x,
              "numpy": lambda x: np.array([x])}


class Discretizer(object):
    def to_index(self, value):
        pass

    def size(self):
        pass

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

class Binning(Discretizer):
    def __init__(self, ranges, nbins):
        self.ranges = ranges
        self.nbins = nbins
        self.bins = []
        self.sizes = tuple()
        for R, b in zip(self.ranges, self.nbins):
            tmp = pd.cut(R, bins=b, retbins=True)[1][1:-1]
            self.bins.append(tmp)
            self.sizes += (b,)

    def to_index(self, value):
        coordinates = []
        for idx, cord_bin in enumerate(self.bins):
            v = np.clip(value[idx], self.ranges[idx][0], self.ranges[idx][1])
            coordinates.append(to_bin(v, cord_bin))
        scalar_index = np.ravel_multi_index(coordinates, self.sizes)
        return scalar_index

    def size(self):
        return np.prod(self.sizes)

class QLearner(object):
    def __init__(self,
                 state_discretization,
                 discrete_actions,
                 alpha=0.2,
                 gamma=0.9,
                 random_action_rate=0.5,
                 random_action_decay_rate=0.99,
                 action_type="scalar"):
        self.state_discretization = state_discretization
        self.discrete_actions = discrete_actions
        self.num_actions = len(discrete_actions)
        self.alpha = alpha
        self.gamma = gamma
        self.random_action_rate = random_action_rate
        self.random_action_decay_rate = random_action_decay_rate
        self.state = 0
        self.action = 0
        num_states = state_discretization.size()
        self.qtable = np.random.uniform(low=-1, high=1, size=(num_states, self.num_actions))
        self.convert_action = CONVERTERS[action_type]

    def set_initial_state(self, state):
        """
        @summary: Sets the initial state and returns an action
        @param state: The initial state
        @returns: The selected action
        """
        self.state = self.state_discretization.to_index(state)
        self.action = self.qtable[self.state].argsort()[-1]
        real_action = self.discrete_actions[self.action]
        return self.convert_action(real_action)

    def move(self, state_prime, reward):
        """
        @summary: Moves to the given state with given reward and returns action
        @param state_prime: The new state
        @param reward: The reward
        @returns: The selected action
        """
        alpha = self.alpha
        gamma = self.gamma
        state = self.state
        action = self.action
        qtable = self.qtable

        state_prime = self.state_discretization.to_index(state_prime)

        choose_random_action = (1 - self.random_action_rate) <= np.random.uniform(0, 1)

        if choose_random_action:
            action_prime = np.random.randint(0, self.num_actions - 1)
        else:
            action_prime = self.qtable[state_prime].argsort()[-1]

        self.random_action_rate *= self.random_action_decay_rate

        qtable[state, action] = (1 - alpha) * qtable[state, action] +\
                                alpha * (reward + gamma * qtable[state_prime, action_prime])

        self.state = state_prime
        self.action = action_prime
        real_action = self.discrete_actions[self.action]

        return self.convert_action(real_action)
