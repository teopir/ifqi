import gym
import math
import numpy as np
from gym.utils import seeding
from .environment import Environment


class Atari(Environment):
    """
    The Atari environment.

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, name='BreakoutDeterministic-v3'):
        self.horizon = 400
        self.gamma = 0.99

        self.env = gym.make(name)
        self.action_space = self.env.action_space
        self.action_space.values = range(self.action_space.n)
        self.observation_space = self.env.observation_space

        # initialize state
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        state = self.env.reset()
        self.env.state = np.array([state, state, state, state])
        return self.get_state()

    def step(self, action):
        current_state = self.get_state()
        new_state = self.env.step(int(action))
        return np.append(current_state[1:], new_state)

    def get_state(self):
        return self.env.state
