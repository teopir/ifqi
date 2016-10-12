import gym
import math
import numpy as np
from .environment import Environment


class CartPole(Environment):
    """
    TODO COSA FA
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, discreteRew=False):
        self.env = gym.make('CartPole-v0')
        self.env.reset()

        # get state info
        self.action_space = self.env.action_space
        self.action_space.values = range(self.action_space.n)
        self.observation_space = self.env.observation_space

        self.horizon = 400
        self.gamma = 0.99
        # self._absorbing = False
        self._atGoal = False
        self._discreteRew = discreteRew
        self._count = 0

    def _reset(self, state=None):
        # self._absorbing = False
        self._atGoal = False
        self._count = 0
        return self.env.reset()

    def _seed(self, seed=None):
        return self.env._seed(seed)

    def _step(self, action):
        self._count += 1

        action = int(np.reshape(action,()))

        nextState, reward, absorbing, info = self.env.step(action)
        # if isinstance(action, int):
        #     nextState, reward, absorbing, info = self.env.step(action)
        # else:
        #     nextState, reward, absorbing, info = self.env.step(int(action[0]))

        x = nextState[0]
        theta = nextState[2]

        if not self._discreteRew:
            theta_rew = (math.cos(theta) -
                         math.cos(self.env.theta_threshold_radians)) / (
                            1. - math.cos(self.env.theta_threshold_radians))
            x_rew = - abs(x / self.env.x_threshold)
            reward = theta_rew + x_rew

        # self._absorbing = absorbing

        if self._count >= self.horizon:
            self._atGoal = True

        return nextState, reward, absorbing, info

    def _getState(self):
        return np.array(self.env.state)
