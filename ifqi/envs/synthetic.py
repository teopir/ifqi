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
        self.action_space = spaces.Box(low=-10, high=10, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        # initialize state
        self.seed()
        self.reset()

    def step(self, action, render=False):
        u = np.clip(action, -100, 100)
        self.state[0] = self.state[0] + 0.1 * self.state[1]
        self.state[1] = np.sqrt(self.state[1] + u)
        self.state[2] = 99 + self.state[2] + np.random.randn() * 5.0

        cost = - self.state[0] ** 2 - action ** 2

        return self.get_state(), cost, False, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None):
        if state is None:
            self.state = np.array([0., 0., 0.])
        else:
            assert len(state) == 3
            self.state = np.array(state)

        return self.get_state()

    def get_state(self):
        return np.array(self.state)
