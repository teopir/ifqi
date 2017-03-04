#
# Copyright (C) 2013, Will Dabney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
from gym import spaces
from gym.utils import seeding
from environment import Environment

import ifqi.utils.spaces as fqispaces
from builtins import range

"""
TODO: to test
"""


class BicycleNew(Environment):
    """Bicycle balancing/riding domain.
    From the paper:
    Learning to Drive a Bicycle using Reinforcement Learning and Shaping.
    Jette Randlov and Preben Alstrom. 1998.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    name = "Bicycle"

    def __init__(self, **kwargs):

        self.horizon = 50000
        self.x_random = True
        self.initial_states = np.zeros((9, 1))
        self.initial_states[:, 0] = np.linspace(-np.pi,np.pi,9)

        self.gamma = 0.98

        ########################################################################
        # Constants
        ########################################################################
        self._noise = kwargs.setdefault('noise', 0.04)

        # omega, omega_dot, theta, theta_dot
        self._state = np.zeros((4,))
        # x_b, y_b, psi
        self._position = np.zeros((3,))
        self._state_range = np.array([[-np.pi * 12. / 180.,
                                          np.pi * 12. / 180.],
                                         [-np.pi * 2. / 180.,
                                          np.pi * 2. / 180.],
                                         [-np.pi, np.pi],
                                         [-np.pi * 80. / 180.,
                                          np.pi * 80. / 180.],
                                         [-np.pi * 2. / 180.,
                                          np.pi * 2. / 180.]])

        self._psi_range = np.array([-np.pi, np.pi])
        self._reward_fall = -1.0

        self._goal_rsqrd = 100.0

        self._goal_loc = np.array([1000., 0.])

        self.dt = 0.01
        self.v =  10./3.6
        self.g = 9.82
        self.d_CM = 0.3
        self.c = 0.66
        self.h = 0.94
        self.M_c = 15.
        self.M_d = 1.7
        self.M_p = 60.
        self.M = self.M_c + self.M_p
        self.r = 0.34
        self.sigma_dot = self.v / self.r
        self.I_bc = (13./3. * self.M_c * self.h**2 + self.M_p * (self.h + self.d_CM)**2)
        self.I_dc = self.M_d * self.r**2
        self.I_dv = 3./2. * self.M_d * self.r**2
        self.I_dl = 0.5 * self.M_d * self.r**2
        self.l = 1.11

        # gym attributes
        self.viewer = None
        high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])  # todo fix
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)

        nactions = 9
        self.action_space = fqispaces.DiscreteValued(range(9), decimals=0)

        # initialize state
        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None):
        self._absorbing = False
        self.goal = False
        if state is None:
            psi = np.random.rand() * 2*np.pi - np.pi
            self._state.fill(0.0)
            self._position.fill(0.0)
            self._position[2] = psi
            return self._getState()
        else:
            psi = float(state)
            self._state.fill(0.0)
            self._position.fill(0.0)
            self._position[2] = psi
            return self._getState()

    def _step(self, action, render=False):
        intAction = int(action)
        assert self.action_space.contains(intAction)
        T = 2. * ((intAction / 3) - 1)  # Torque on handle bars
        d = 0.02 * ((intAction % 3) - 1)  # Displacement of center of mass (in meters)
        w = (np.random.random()-0.5) * self._noise # Noise between [-0.02, 0.02] meters
        # w = np.random.uniform(low=-self._noise, high=self._noise)

        omega, omega_dot,  theta, theta_dot = tuple(self._state)
        x_b, y_b, psi = tuple(self._position)

        phi = omega + np.arctan(d + w) / self.h
        invr_f = np.abs(np.sin(theta)) / self.l
        invr_b = np.abs(np.tan(theta)) / self.l
        invr_CM = 0.
        if not np.isclose(theta, 0., atol=1e-6):
            invr_CM = 1. / np.sqrt((self.l - self.c)**2 + (1. / invr_b)**2)

        omega_t1 = omega + self.dt * omega_dot
        tmp1 = self.M * self.h * self.g * np.sin(phi)
        tmp2 = self.I_dc * self.sigma_dot * theta_dot
        tmp3 = self.M_d * self.r * (invr_f + invr_b) + self.M *self.h * invr_CM
        omega_dot_t1 = omega_dot + self.dt * (
            1. / self.I_bc * (tmp1
                              - np.cos(phi)
                              * (tmp2 + np.sign(theta) * self.v**2 * tmp3)
                              )
        )

        theta_t1 = theta + self.dt * theta_dot
        if np.abs(theta_t1) > 80./180. * np.pi:
            theta_t1 = np.sign(theta_t1) * 80./180. * np.pi
            theta_dot_t1 = 0.
        else:
            theta_dot_t1 = theta_dot + self.dt * (
            (T - self.I_dv * self.sigma_dot * omega_dot) / self.I_dl)

        x_b_t1 = x_b + self.dt * self.v * np.cos(psi)
        y_b_t1 = y_b + self.dt * self.v * np.sin(psi)

        psi_t1 = psi + self.dt * np.sign(theta) * self.v * invr_b

        # check reward and terminal condition
        if np.abs(omega_t1) > 12./180. * np.pi:
            self._absorbing = True
            reward = -1.
        else:
            self._absorbing = False
            reward = 0.0
            # reward = 0.1 * (self._angleWrapPi(psi) - self._angleWrapPi(
            #     psi_t1)) if self._navigate else 0.0

        self._state = np.array([omega_t1, omega_dot_t1, theta_t1, theta_dot_t1])
        self._position = np.array([x_b_t1, y_b_t1, psi_t1])

        return self._getState(), \
               reward, self._absorbing, \
               {"goal": 1. if self._isAtGoal() else 0.,
                "dist": np.linalg.norm(self._position[:2] - self._goal_loc, 2),
                "pos_x": float(self._position[:1]),
                "pos_y": float(self._position[1:2])}

    def _unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def _angle_between(self, v1, v2):

        v1_u = self._unit_vector(v1)
        v2_u = self._unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def _isAtGoal(self):
        # Anywhere in the goal radius
        return False

    def _getState(self):
        omega, omega_dot, theta, theta_dot = tuple(self._state)
        x_b, y_b, psi = tuple(self._position)
        x_f = x_b + np.cos(psi) * self.l
        y_f = y_b + np.sin(psi) * self.l
        goal_angle = self._angleWrapPi(
            self._angle_between(
                self._goal_loc - np.array([x_b,y_b]),
                np.array([x_f - x_b, y_f - y_b])
            )
        )
        """ modified to follow Ernst paper"""
        return np.array([omega, omega_dot, theta, theta_dot, psi])



    def _angleWrapPi(self, x):
        while (x < -np.pi):
            x += 2.0 * np.pi
        while (x > np.pi):
            x -= 2.0 * np.pi
        return x
