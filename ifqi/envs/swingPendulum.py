# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:25:48 2016

@author: samuele

Pendulum as described in Reinforcement Learning in Continuous Time and Space
"""


import numpy as np
from gym import spaces
from gym.utils import seeding

import ifqi.utils.spaces as fqispaces
from .environment import Environment


class SwingPendulum(Environment):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, **kwargs):

        self.initial_states = np.zeros((21, 2))
        self.initial_states[:, 0] = np.linspace(-np.pi, np.pi, 21)

        self.gamma = 0.98

        self._m = 1.
        self._l = 1.
        self._g = 9.8
        self._mu = 0.01
        self._dt = 0.02
        self.horizon = int(20. / self._dt)
        self.time_up = int(10. / self._dt)
        self.time_over_rotated = int(10. / self._dt)

        # gym attributes
        self.viewer = None
        high = np.array([np.inf, np.inf])
        self.observation_space = spaces.Box(low=-high, high=high)
        self.action_space = fqispaces.DiscreteValued([-5, 5], decimals=5)

        # initialize state
        self.seed()
        self.reset()

    def _step(self, action, render=False):
        u = action[0]
        theta, theta_dot = tuple(self.get_state())

        theta_ddot = (-self._dt * theta_dot + self._m * self._l * self._g *
                      np.sin(theta_dot) + u)

        # bound theta_dot
        theta_dot_temp = theta_dot + theta_ddot
        if theta_dot_temp > np.pi / self._dt:
            theta_dot_temp = np.pi / self._dt
        if theta_dot_temp < -np.pi / self._dt:
            theta_dot_temp = -np.pi / self._dt

        theta_dot = theta_dot_temp
        theta += theta_dot * self._dt

        # adjust Theta
        if theta > np.pi:
            theta -= 2 * np.pi
        if theta < -np.pi:
            theta += 2 * np.pi

        self._state = np.array([theta, theta_dot])
        reward = np.cos(theta)

        """if abs(theta) > 5 * np.pi:
            self.t_over_rotated += 1
        else:
            self.time_over_rotated = 0"""

        if abs(theta) < np.pi / 4:
            self.t_up += 1
        else:
            self.t_up = 0

        goal = 0
        if self.t_up > self.time_up:
            goal = 1
            self.done = True

        """if self.time_over_rotated > self.time_over_rotated:
            self.done = False"""

        return self.get_state(), reward, self.done, {"goal":self.goal}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None):
        if state is None:
            theta = self.np_random.uniform(low=-np.pi, high=np.pi)
            self._state = np.array([theta, 0.])
        else:
            self._state = np.array(state).ravel()

        self.done = False
        self.t_up = 0
        self.t_over_rotated = 0

        return self.get_state()

    def get_state(self):
        return self._state
