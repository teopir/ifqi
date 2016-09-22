"""
Here there is a version of LQG1D with discrete actions.
Example of usage (write this script in examples)

from context import *
from ifqi.ifqi.envs.lqg1dDiscrete import LQG1DDiscrete
import numpy as np
env = LQG1DDiscrete(np.linspace(-10,10, 50))

for _ in xrange(100):    
    env._step(np.random.randint(50))
    env.render()
"""

from lqg1d import LQG1D

class LQG1DDiscrete(LQG1D):
    
    def __init__(self, actions, **args):
        """
        This function initialize the Discretized version of LQG1D
            params:
                actions: a list (or np array) with all possible values that could be generated 
                **args: args to pass to LQG1D constructor
        """
        self.actions = actions
        if isinstance(actions, list):
            self.nActions = len(actions)
        else:
            try:
                self.nActions = actions.shape[0]
            except:
                raise Exception("Please pass here a list or a numpy array.")
        LQG1D.__init__(self,**args)
    
    def _step(self, action):
        a = self.actions[action]
        return LQG1D._step(self,a)
	
	
