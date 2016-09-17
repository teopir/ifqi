import numpy as np

from environment import Environment

class InvPendulum(Environment):
    """
    The Inverted Pendulum environment.
    
    """
    def __init__(self):
        """
        Constructor.
        
        """
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
        # Time_step
        self._dt = 0.1
        # Discount factor
        self.gamma=.95
    
    def _step(self, u, render=False):
        act = u * 50. - 50.
        n_u = act  +  2 * self._noise * np.random.rand() - self._noise
        
        a = self._g * np.sin(self._theta) - self._alpha * self._m * self._l * self._theta_dot**2 * np.sin(2 * self._theta) / 2. - self._alpha * np.cos(self._theta) * n_u
        b = 4. *self._l / 3. - self._alpha * self._m * self._l * (np.cos(self._theta)) ** 2
        
        theta_ddot = a/b
    
        self._theta_dot = self._theta_dot + self._dt * theta_ddot
        self._theta = self._theta + self._dt * self._theta_dot
        
        if(np.abs(self._theta) > self._angleMax):
            self._absorbing = True
            return -1
        else:
            return 0

    def _reset(self, state=None):
        self._absorbing=False
        self._theta = 0
        self._theta_dot = 0
        
    def _getState(self):
        return [self._theta, self._theta_dot]
        
    def evaluate(self, fqi, expReplay=False, render=False):
        """
        This function evaluates the regressor in the provided object parameter.
        This way of evaluation is just one of many possible ones.
        Params:
            fqi (object): an object containing the trained regressor
            expReplay (bool): flag indicating whether to do experience replay
            render (bool): flag indicating whether to render visualize behavior of the agent
        Returns:
            ...
        
        """
        pass