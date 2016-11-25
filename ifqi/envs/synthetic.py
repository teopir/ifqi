import numpy as np
from gym import spaces
from gym.utils import seeding
from gym.spaces import prng

from .environment import Environment


class SyntheticToyFS(Environment):
    def __init__(self):
        self.horizon = 10
        self.gamma = 0.99
        # gym attributes
        self.viewer = None
        high = np.array([np.inf] * 3)
        self.action_space = spaces.Box(low=-100, high=100, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        # initialize state
        self.seed()
        self.reset()

    def step(self, action, render=False):
        self.state[0] = self.state[0] * np.exp(0.6 * self.state[1]) + 1.0 / action
        self.state[1] = self.state[1] ** 2 + np.random.rand(1)
        self.state[2] = self.state[2] + np.random.rand(1) * 5.0

        cost = - self.state[0] ** 2 - action ** 2

        return self.get_state(), -np.asscalar(cost), False, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None):
        if state is None:
            self.state = np.array([0, 0, 0])
        else:
            assert len(state) == 3
            self.state = np.array(state)

        return self.get_state()

    def get_state(self):
        return self.state
