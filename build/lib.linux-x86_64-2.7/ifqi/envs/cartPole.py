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

    def __init__(self):
        self.horizon = 400
        self.gamma = 0.99

        self.env = gym.make('CartPole-v0')
        self.reset()

        self.action_space = self.env.action_space
        self.action_space.values = range(self.action_space.n)
        self.observation_space = self.env.observation_space

    def reset(self, state=None):
        if state is None:
            return self.env.reset()
        else:
            self.env.state = state
            return self.get_state()

    def step(self, action):
        return self.env.step(int(action))

    def get_state(self):
        return self.env.state
