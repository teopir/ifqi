import gym
import math
import numpy as np
from gym.utils import seeding
from .environment import Environment


class LunarLander(Environment):
    """
    The CartPole environment.

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 100
    }

    def __init__(self):
        self.horizon = 10000
        self.gamma = 1.

        self.env = gym.make('LunarLander-v2')
        self.action_space = self.env.action_space
        self.action_space.values = range(self.action_space.n)
        self.observation_space = self.env.observation_space

        # initialize state
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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

    def render(self, mode='human', close=False):
        self.env.render()
