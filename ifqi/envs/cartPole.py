import gym

from gymEnv import GymEnv

class CartPole(GymEnv):
    def __init__(self):
        self._env = gym.make('CartPole-v0')
        self._env.reset()
        self.state_dim = 4
        self.action_dim = 1        
        self.n_actions = 2  
        self.gamma = 0.99
        self._absorbing = False
        self._atGoal = False
        self._nextState = [0., 0., 0., 0.]
        
    def _reset(self):
        self._nextState = [0., 0., 0., 0.]
        self._absorbing = False
        self._atGoal = False
        self._env.reset()

    def _step(self, action, render=False):
        if render:
            self._env.render()
        nextState, reward, absorbing, info = self._env.step(action) 
        self._nextState = nextState
        self._absorbing = absorbing

        return reward
        
    def _getState(self):
        return self._nextState
        
    def evaluate(self, fqi, expReplay=False, render=False):
        """
        This function evaluates the regressor in the provided object parameter.
        This way of evaluation is just one of many possible ones.
        Params:
            fqi (object): an object containing the trained regressor.
        Returns:
        
        """
        self._reset()
        J, step, goal, sast, r = self._runEpisode(fqi, expReplay, render)
               
        return (J, step, goal, sast, r)