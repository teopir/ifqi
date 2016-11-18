# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:25:48 2016

@author: samuele

Pendulum as described in Reinforcement Learning in Continuous Time and Space
"""

"""
TODO: to test
"""

import numpy as np
from gym import spaces
from gym.utils import seeding

import ifqi.utils.spaces as fqispaces
from .environment import Environment


class SwingPendulum(Environment):
    name = "swingUpPendulum"
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, **kwargs):
        #TODO: check actual horizon
        self.horizon = 100
        self.gamma = 0.9

        self._m = 1.
        self._l = 1.
        self._g = 9.8
        self._mu = 0.01

        self._dt = 0.02
        self._overTime = 0
        self._upTime = 0

        self._atGoal = False
        self._abs = False

        self._previousTheta = 0

        self._overRotated = False
        self._cumulatedRotation = 0
        self._overRotatedTime = 0

        self.max_angle = np.inf
        self.max_velocity = np.inf

        # gym attributes
        self.viewer = None
        high = np.array([self.max_angle, self.max_velocity])
        self.observation_space = spaces.Box(low=-high, high=high)

        nactions = 3
        actions = [u * 10.0 / (nactions-1) - 5.0 for u in range(nactions)]
        self.action_space = fqispaces.DiscreteValued(actions, decimals=5)

        # initialize state
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self,state=None):
        return self._reset(state=state)

    def _reset(self, state=None):
        if state is None:
            theta = np.random.uniform(-np.pi, np.pi)
            self.state = [theta, 0.]
        else:
            self.state = [state[0], state[1]]
        self._overTime = 0
        self._upTime = 0

        self._atGoal = False
        self._abs = False
        self._previousTheta = 0
        self._overRotated = False
        self._overRotatedTime = 0
        self._cumulatedRotation = 0
        return self._getState()

    def _step(self, action, render=False):

        # u = action[0] / 11. * 10. - 5.

        u = np.reshape(action,())

        theta, theta_dot = tuple(self.state)

        # theta_ddot = (- self.mu * theta_dot + self.m * self.l * self.g *
        # np.sin(theta_dot) + u)/ (self.m * self.l * self.l)
        theta_ddot = (- self._dt * theta_dot + self._m * self._l * self._g *
                      np.sin(theta_dot) + u)  # / (self.m * self.l * self.l)

        # bund theta_dot
        theta_dot_temp = theta_dot + theta_ddot
        if theta_dot_temp > np.pi / self._dt:
            theta_dot_temp = np.pi / self._dt
        if theta_dot_temp < -np.pi / self._dt:
            theta_dot_temp = -np.pi / self._dt

        theta_dot = theta_dot_temp  # theta_ddot #* self.dt
        theta += theta_dot * self._dt

        # Adjust Theta
        if theta > np.pi:
            theta -= 2 * np.pi
        if theta < -np.pi:
            theta += 2 * np.pi

        self.state = [theta, theta_dot]

        """signAngleDifference = np.arctan2(np.sin(theta - self.previousTheta),
                                            np.sin(theta - self.previousTheta))
        self.cumulatedRotation += signAngleDifference

        if (not self.overRotated and
            abs(self.cumulatedRotation) > 5.0 * np.pi):
            self.overRotated = True
        if (self.overRotated):
            self.overRotatedTime += 1

        self.abs = self.overRotated and (self.overRotatedTime > 1.0 / self.dt)
        """

        """
        if not self.overRotated:
            return np.cos(theta)
        else:
            return -1"""
        reward = np.cos(theta)

        return self._getState(), reward, False, {}

    def _getState(self):
        return np.array(self.state)
