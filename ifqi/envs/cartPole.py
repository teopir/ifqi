import gym
import math
import numpy as np

from gym import wrappers
from gym import envs
from gym.utils import seeding
from .environment import Environment


class CartPole(Environment):
    """
    The CartPole environment.

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, continuosReward = False):

        self.x_random = True
        self.gamma = 1.

        self.env = gym.make('CartPole-v0')

        self.horizon = envs.registry.env_specs["CartPole-v0"].tags['wrapper_config.TimeLimit.max_episode_steps']

        self.action_space = self.env.action_space
        self.action_space.values = range(self.action_space.n)
        self.observation_space = self.env.observation_space

        self.continuousReward = continuosReward
        # initialize state
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None):
        if state is None:
            self.env.reset()
            if self.x_random:
                x = np.random.rand() * 2 * self.env.x_threshold - self.env.x_threshold
                dx = np.random.rand() * 7. - 3.5
                th = np.random.rand() * 12 * 2 * math.pi / 360 - 12 * math.pi / 360
                th *= 0.95
                self.env.state[0] = x
                self.env.state[1] = dx
                dth = np.random.rand() * 6. - 3.
                self.env.state[3] = dth
            return self.get_state()
        else:
            self.env.state = state
            return self.get_state()

    def step(self, action):
        ret = self.env.step(int(action))
        ret = list(ret)
        if self.continuousReward:
            ret[1] = np.cos(ret[0][2] * 15.) - 10.
        ret = tuple(ret)
        return ret

    def render(self, mode='human', close=False):
        return
        self.env.render()


    def get_state(self):
        return self.env.state
