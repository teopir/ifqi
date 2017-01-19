import gym
import numpy as np
from gym import spaces


class SyntheticToyFS(gym.Env):
    def __init__(self):
        self.horizon = 10
        self.gamma = 0.99
        # gym attributes
        self.viewer = None
        high = np.array([np.inf] * 4)
        high_a = np.array([10, 10])
        self.action_space = spaces.Box(low=-high_a, high=high_a)
        self.observation_space = spaces.Box(low=-high, high=high)

        # initialize state
        self.seed()
        self.reset()

    def step(self, action, render=False):
        # action 1 is useless
        u = np.clip(action[0], -100, 100)
        self.state[0] = self.state[0] + 0.1 * self.state[2]
        self.state[1] = 99 + self.state[1] * (-1 + 2 * np.random.randn()) * 5.0
        x = self.state[2] + 0.5 * u
        self.state[2] = np.power(x, 1.0 / 3.0) if x > 0 else -np.power(-x, 1.0 / 3.0)
        self.state[3] = (-8 + self.state[2]) * np.random.randn()

        cost = - self.state[0] ** 2 - u

        return self.get_state(), cost, False, {}

    def reset(self, state=None):
        if state is None:
            self.state = np.array([0., 0., 0., -1.])
        else:
            assert len(state) == 3
            self.state = np.array(state)

        return self.get_state()

    def get_state(self):
        return np.array(self.state)
