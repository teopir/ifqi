import gym
import math
import numpy as np
from .environment import Environment


class CartPole(Environment):
    """
    The CartPole environment.

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, discreteRew=False):
        self.env = gym.make('CartPole-v0')
        self._reset()

        self.action_space = self.env.action_space
        self.action_space.values = range(self.action_space.n)
        self.observation_space = self.env.observation_space

        self.horizon = 400
        self.gamma = 0.99

    def _reset(self):
        self._count = 0
        return self.env.reset()

    def _step(self, action):
        self._count += 1
        action = int(np.reshape(action, ()))

        return self.env.step(action)

    def _get_state(self):
        return np.array(self.env.state)
