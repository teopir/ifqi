import numpy as np
from scipy.integrate import odeint

from environment import Environment


class CarOnHill(Environment):
    """
    The Car On Hill environment as presented in:
    "Tree-Based Batch Mode Reinforcement Learning, D. Ernst et. al."

    """
    def __init__(self):
        # Properties
        self.stateDim = 2
        self.actionDim = 1
        self.nStates = 0
        self.nActions = 2
        self.horizon = 100.
        # State
        self._position = -0.5
        self._velocity = 0.
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

    def _step(self, u, render=False):
        stateAction = np.append([self._position, self._velocity], u)
        newState = odeint(self._dpds, stateAction, [0, self._dt])

        newState = newState[-1]
        self._position = newState[0]
        self._velocity = newState[1]

        if(self._position < -1 or np.abs(self._velocity) > 3):
            self._absorbing = True
            return -1
        elif(self._position > 1 and np.abs(self._velocity) <= 3):
            self._absorbing = True
            self._atGoal = True
            return 1
        else:
            return 0

    def _reset(self, state=[-0.5, 0]):
        self._absorbing = False
        self._atGoal = False
        self._position = state[0]
        self._velocity = state[1]

    def _getState(self):
        return [self._position, self._velocity]

    def _dpds(self, stateAction, t):
        position = stateAction[0]
        velocity = stateAction[1]
        action = stateAction[-1]

        if position < 0.:
            diffHill = 2 * position + 1
            diff2Hill = 2
        else:
            diffHill = 1 / ((1 + 5 * position ** 2) ** 1.5)
            diff2Hill = (-15 * position) / ((1 + 5 * position ** 2) ** 2.5)

        u = -4. if action == 0 else 4.

        dp = velocity
        ds = (u - self._g * self._m * diffHill - velocity ** 2 * self._m *
              diffHill * diff2Hill) / (self._m * (1 + diffHill ** 2))

        return (dp, ds, 0.)

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
            289 different states

        """
        discRewards = np.zeros((289))

        counter = 0
        for i in range(-8, 9):
            for j in range(-8, 9):
                position = 0.125 * i
                velocity = 0.375 * j

                self._reset([position, velocity])
                J = self.runEpisode(fqi, expReplay, render)[0]

                discRewards[counter] = J
                counter += 1

        return [np.mean(discRewards)]
