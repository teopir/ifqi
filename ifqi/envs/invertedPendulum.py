import gym
import numpy as np
from gym import spaces

import ifqi.utils.spaces as fqispaces


class InvPendulum(gym.Env):
    """
    The Inverted Pendulum environment.

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self):
        self.horizon = 400
        self.gamma = .95

        self._theta = 0.
        self._theta_dot = 0.

        self._g = 9.8
        self._m = 2.
        self._M = 8.
        self._l = .5
        self._alpha = 1. / (self._m + self._M)
        self._noise = 10.
        self._angle_max = np.pi / 2.
        self.max_velocity = np.inf
        self._dt = 0.1

        # gym attributes
        self.viewer = None
        high = np.array([self._angle_max, self.max_velocity])
        self.observation_space = spaces.Box(low=-high, high=high)
        self.action_space = fqispaces.DiscreteValued([-50, 0, 50], decimals=0)

        # initialize state
        self.seed()
        self.reset()

    def step(self, u):
        n_u = u + 2 * self._noise * np.random.rand() - self._noise

        a = self._g * np.sin(self._theta) - self._alpha * self._m * self._l * \
            (self._theta_dot ** 2) * np.sin(2 * self._theta) / 2. \
            - self._alpha * np.cos(self._theta) * n_u
        b = 4. * self._l / 3. \
            - self._alpha * self._m * self._l * (np.cos(self._theta)) ** 2

        theta_ddot = a / b

        self._theta_dot = self._theta_dot + self._dt * theta_ddot
        self._theta = self._theta + self._dt * self._theta_dot

        reward = 0
        if np.abs(self._theta) > self._angle_max:
            self._absorbing = True
            reward = -1

        return self.get_state(), reward, self._absorbing, {}

    def reset(self, state=None):
        self._absorbing = False
        if state is None:
            self.state = np.array([0., 0.])
        else:
            self.state = state

        return self.get_state()

    def get_state(self):
        return self.state
