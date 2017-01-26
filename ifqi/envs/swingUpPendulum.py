import gym
import math
import numpy as np
from gym import envs
from gym.utils import seeding
from .environment import Environment

from gym import wrappers

class SwingUpPendulum(Environment):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 100
    }

    def __init__(self):
        self.gamma = 1.
        self.metric = "average"
        self.state_dim = 2
        self.action_dim = 1
        self.env = gym.make('Pendulum-v0')
        self.horizon = 200 #envs.registry.env_specs["Pendulum-v0"].tags['wrapper_config.TimeLimit.max_episode_steps']

        #self.env = wrappers.Monitor(self.env, './monitor',force=True)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.tup = 0

        # initialize state
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None):
        self.tup = 0
        if state is None:
            return self.env.reset()
        else:
            self.env.state = state
            return self.get_state()

    def step(self, action):
        ret = self.env.step(action)
        ret = list(ret)
        if abs(ret[0][0]) < np.pi / 4.:
            self.tup += self.env.dt
        #ret[1] = np.cos(ret[0][0])
        ret[-1] = {"t_up": self.tup}
        ret = tuple(ret)
        return ret

    def get_state(self):
        return self.env.state

    def render(self, mode='human', close=False):
        self.env.render()