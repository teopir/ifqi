import gym
import math
import numpy as np

from gym import wrappers
from gym import envs
from gym.utils import seeding
from .environment import Environment


class AcrobotGym(Environment):
    """
    The CartPole environment.

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self):

        self.x_random = True
        self.gamma = 1.

        self.env = gym.make('Acrobot-v1')

        self.horizon = envs.registry.env_specs["Acrobot-v1"].tags['wrapper_config.TimeLimit.max_episode_steps']
        print "horizon", self.horizon
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
            self.env.reset()
            if self.x_random:
                self.env.state[0] = np.random.rand()  * np.pi
                self.env.state[1] = np.random.rand() * np.pi
                self.env.state[2] = np.random.rand() * 8.
                self.env.state[3] = np.random.rand() *16

            return self.get_state()
        else:
            self.env.state = state
            return self.get_state()

    def step(self, action):
        ret = self.env.step(int(action))
        return ret

    def render(self, mode='human', close=False):
        self.env.render()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self):
        return self.env._get_ob()