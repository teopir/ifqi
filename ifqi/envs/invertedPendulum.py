import numpy as np
from gym import spaces
from gym.utils import seeding

import ifqi.utils.spaces as fqispaces
from .environment import Environment


class InvPendulum(Environment):
    """
    The Inverted Pendulum environment.

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    # TODO dove e' stato preso???

    def __init__(self):
        """
        Constructor.

        """
        # Properties
        # self.stateDim = 2
        # self.actionDim = 1
        # self.nStates = 0
        # self.nActions = 3
        # self.horizon = 100.
        # State
        self._theta = 0.
        self._theta_dot = 0.
        # End episode
        self._absorbing = False
        self._atGoal = False
        # Constants
        self._g = 9.8
        self._m = 2.
        self._M = 8.
        self._l = .5
        self._alpha = 1. / (self._m + self._M)
        self._noise = 10.
        self._angleMax = np.pi / 2.
        self.max_velocity = np.inf
        # Time_step
        self._dt = 0.1
        # Discount factor
        self.gamma = .95
        #TODO: check horizon of InvertedPendulum
        self.horizon = 400
        # gym attributes
        self.viewer = None
        high = np.array([self._angleMax, self.max_velocity])
        self.observation_space = spaces.Box(low=-high, high=high)
        self.action_space = fqispaces.DiscreteValued([-50, 0, 50], decimals=0)

        # initialize state
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, u, render=False):
        # u = np.clip(u, -self.max_action, self.max_action)
        n_u = u + 2 * self._noise * np.random.rand() - self._noise

        a = self._g * np.sin(self._theta)\
            - self._alpha * self._m * self._l * (self._theta_dot ** 2) * np.sin(2 * self._theta) / 2. \
            - self._alpha * np.cos(self._theta) * n_u
        b = 4. * self._l / 3. \
            - self._alpha * self._m * self._l * (np.cos(self._theta)) ** 2

        theta_ddot = a / b

        self._theta_dot = self._theta_dot + self._dt * theta_ddot
        self._theta = self._theta + self._dt * self._theta_dot

        reward = 0
        if np.abs(self._theta) > self._angleMax:
            self._absorbing = True
            reward = -1

        return self._getState(), reward, self._absorbing, {}

    def _reset(self, state=None):
        self._absorbing = False
        if state is None:
            self._theta = 0
            self._theta_dot = 0
        else:
            self._theta = state[0]
            self._theta_dot = state[1]
        return self._getState()

    def _getState(self):
        return np.array([self._theta, self._theta_dot])
