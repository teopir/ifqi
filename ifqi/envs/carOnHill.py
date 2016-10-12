import numpy as np
from builtins import range
from gym import spaces
from gym.utils import seeding
from scipy.integrate import odeint

import ifqi.utils.spaces as fqispaces
from .environment import Environment


class CarOnHill(Environment):
    """
    The Car On Hill environment as presented in:
    "Tree-Based Batch Mode Reinforcement Learning, D. Ernst et. al."

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self):
        # Properties
        # self.stateDim = 2
        # self.actionDim = 1
        # self.nStates = 0
        # self.nActions = 2
        self.horizon = 100.
        # State
        self._position = None
        self._velocity = None
        self.max_pos = 1.
        self.max_velocity = 3.
        self.max_action = 4.
        # End episode
        self._absorbing = False
        self._atGoal = False
        # Constants
        self._g = 9.81
        self._m = 1
        # Time_step
        self._dt = .1
        # Discount factor
        self.gamma = .95

        # gym attributes
        self.viewer = None
        high = np.array([self.max_pos, self.max_velocity])
        self.observation_space = spaces.Box(low=-high, high=high)
        self.action_space = fqispaces.DiscreteValued([-4., 4.], decimals=0)

        # initialize state
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, u, render=False):
        # u = np.clip(u, -self.max_action, self.max_action)
        
        stateAction = np.array([self._position, self._velocity, u])
        newState = odeint(self._dpds, stateAction, [0, self._dt])

        newState = newState[-1]
        self._position = newState[0]
        self._velocity = newState[1]

        if self._position < -self.max_pos or \
                        np.abs(self._velocity) > self.max_velocity:
            self._absorbing = True
            reward = -1
        elif self._position > self.max_pos and \
                        np.abs(self._velocity) <= self.max_velocity:
            self._absorbing = True
            self._atGoal = True
            reward = 1
        else:
            reward = 0
        return self._getState(), reward, self._absorbing, {}

    def _reset(self, state=None):
        self._absorbing = False
        self._atGoal = False
        if state is None:
            self._position = -0.5
            self._velocity = 0.0
        else:
            self._position = state[0]
            self._velocity = state[1]
        return self._getState()

    def _getState(self):
        return np.array([self._position, self._velocity])

    def _dpds(self, stateAction, t):
        position = stateAction[0]
        velocity = stateAction[1]
        u = stateAction[-1]

        if position < 0.:
            diffHill = 2 * position + 1
            diff2Hill = 2
        else:
            diffHill = 1 / ((1 + 5 * position ** 2) ** 1.5)
            diff2Hill = (-15 * position) / ((1 + 5 * position ** 2) ** 2.5)

        dp = velocity
        ds = (u - self._g * self._m * diffHill - velocity ** 2 * self._m *
              diffHill * diff2Hill) / (self._m * (1 + diffHill ** 2))

        return dp, ds, 0.

    def evaluate(self, policy, nbEpisodes=1, metric='discounted',
                 initialState=None, render=False):
        """
        This function evaluates policy starting from 289 discretized states.
        For each position n_episodes are performed,
        Params:
            policy (object): a policy object (method drawAction is expected)
            nbEpisodes (int): the number of episodes to execute
            metric (string): the evaluation metric ['discounted', 'average']
            initialState: NOT used
            render (bool): flag indicating whether to visualize the behavior of
                            the policy
        Returns:
            metric (float): the selected evaluation metric
            confidence (float): 95% confidence level for the provided metric

        """
        nstates = 289
        values = np.zeros(nstates)
        counter = 0
        for i in range(-8, 9):
            for j in range(-8, 9):
                position = 0.125 * i
                velocity = 0.375 * j

                x0 = [position, velocity]

                values[counter] = \
                    super(CarOnHill, self).evaluate(policy, nbEpisodes, metric, x0, render)[0]
                counter += 1

        return values.mean(), 2 * values.std() / np.sqrt(nstates)
