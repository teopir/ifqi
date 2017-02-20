import gym
import math
import numpy as np
from gym import envs
from gym.utils import seeding
from .environment import Environment
import time
from gym import wrappers

class SwingUpPendulum(Environment):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 100
    }

    def __init__(self, cont_reward=True):
        self.gamma = 1.
        self.metric = "average"
        self.state_dim = 2
        self.action_dim = 1
        self.env = gym.make('Pendulum-v0')
        self.horizon = envs.registry.env_specs["Pendulum-v0"].tags['wrapper_config.TimeLimit.max_episode_steps']

        #self.env = wrappers.Monitor(self.env, './monitor',force=True)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.tup = 0

        # initialize state
        self.cont_reward = cont_reward
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
        if self.cont_reward:
            ret = list(ret)
            if abs(ret[0][0] - 1) < 0.1:
                self.tup += self.env.dt
            reward = ret[0][0]
            ret[-1] = {"t_up": self.tup}
            ret = tuple(ret)
        return ret

    def get_state(self):
        return self.env.state

    def render(self, mode='human', close=False):
        self.env.render()