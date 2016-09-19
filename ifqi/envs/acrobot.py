import numpy as np
from numpy.random import uniform
from scipy.integrate import odeint

from environment import Environment


class Acrobot(Environment):
    """
    The Acrobot environment as presented in:
    "Tree-Based Batch Mode Reinforcement Learning, D. Ernst et. al."

    """
    def __init__(self):
        # Properties
        self.stateDim = 4
        self.actionDim = 1
        self.nStates = 0
        self.nActions = 2
        self.horizon = 100.
        # State
        self._theta1 = uniform(-np.pi + 1, np.pi - 1)
        self._theta2 = self._dTheta1 = self._dTheta2 = 0.
        # End episode
        self._absorbing = False
        self._atGoal = False
        # Constants
        self._g = 9.81
        self._M1 = self._M2 = 1
        self._L1 = self._L2 = 1
        self._mu1 = self._mu2 = .01
        # Time_step
        self._dt = .1
        # Discount factor
        self.gamma = .95

    def _step(self, u, render=False):
        stateAction = np.append([self._theta1,
                                 self._theta2,
                                 self._dTheta1,
                                 self._dTheta2], u)
        newState = odeint(self._dpds,
                          stateAction,
                          [0, self._dt],
                          rtol=1e-5,
                          atol=1e-5,
                          mxstep=2000)

        newState = newState[-1]
        self._theta1 = newState[0]
        self._theta2 = newState[1]
        self._dTheta1 = newState[2]
        self._dTheta2 = newState[3]

        k = round((self._theta1 - np.pi) / (2 * np.pi))
        x = np.array([self._theta1,
                      self._theta2,
                      self._dTheta1,
                      self._dTheta2])
        o = np.array([2 * k * np.pi + np.pi, 0., 0., 0.])
        d = np.linalg.norm(x - o)

        self._theta1 = self._wrap2pi(self._theta1)
        self._theta2 = self._wrap2pi(self._theta2)
        if(d < 1):
            self._absorbing = True
            self._atGoal = True
            return 1 - d
        else:
            return 0

    def _reset(self, state=[-2, 0., 0., 0.]):
        self._absorbing = False
        self._atGoal = False
        self._theta1 = self._wrap2pi(state[0])
        self._theta2 = self._wrap2pi(state[1])
        self._dTheta1 = state[2]
        self._dTheta2 = state[3]

    def _getState(self):
        return [self._theta1, self._theta2, self._dTheta1, self._dTheta2]

    def _dpds(self, stateAction, t):
        theta1 = stateAction[0]
        theta2 = stateAction[1]
        dTheta1 = stateAction[2]
        dTheta2 = stateAction[3]
        action = stateAction[-1]

        d11 = self._M1 * self._L1 * self._L1 + self._M2 * \
            (self._L1 * self._L1 + self._L2 * self._L2 + 2 * self._L1 *
             self._L2 * np.cos(theta2))
        d22 = self._M2 * self._L2 * self._L2
        d12 = self._M2 * (self._L2 * self._L2 + self._L1 * self._L2 *
                          np.cos(theta2))
        c1 = -self._M2 * self._L1 * self._L2 * dTheta2 * \
            (2 * dTheta1 + dTheta2 * np.sin(theta2))
        c2 = self._M2 * self._L1 * self._L2 * dTheta1 * dTheta1 * \
            np.sin(theta2)
        phi1 = (self._M1 * self._L1 + self._M2 * self._L1) * self._g * \
            np.sin(theta1) + self._M2 * self._L2 * self._g * \
            np.sin(theta1 + theta2)
        phi2 = self._M2 * self._L2 * self._g * np.sin(theta1 + theta2)

        u = -5. if action == 0 else 5.

        diffTheta1 = dTheta1
        diffTheta2 = dTheta2
        d12d22 = d12 / d22
        diffDiffTheta1 = (-self._mu1 * dTheta1 - d12d22 * u + d12d22 *
                          self._mu2 * dTheta2 + d12d22 * c2 + d12d22 * phi2 -
                          c1 - phi1) / (d11 - (d12d22 * d12))
        diffDiffTheta2 = (u - self._mu2 * dTheta2 - d12 * diffDiffTheta1 -
                          c2 - phi2) / d22

        return (diffTheta1, diffTheta2, diffDiffTheta1, diffDiffTheta2, 0.)

    def _wrap2pi(self, value):
        tmp = value - -np.pi
        width = 2 * np.pi
        tmp -= width * np.floor(tmp / width)

        return tmp + -np.pi

    def evaluate(self, fqi, expReplay=False, render=False):
        """
        This function evaluates the regressor in the provided object parameter.
        This way of evaluation is just one of many possible ones.
        Params:
            fqi (object): an object containing the trained regressor
            expReplay (bool): flag indicating whether to do experience replay
            render (bool): flag indicating whether to render visualize behavior
                           of the agent
        Returns:
            a numpy array containing the average score obtained starting from
            41 different states

        """
        states = np.linspace(-2, 2, 41)

        discRewards = np.zeros((states.size))

        counter = 0
        for theta1 in states:
            self._reset([theta1, 0., 0., 0.])
            J = self.runEpisode(fqi, expReplay, render)[0]

            discRewards[counter] = J
            counter += 1

        return [np.mean(discRewards)]
