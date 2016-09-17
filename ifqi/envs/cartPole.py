import gym

from gymEnv import GymEnv

class CartPole(GymEnv):
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.env.reset()
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
        self.env.reset()

    def _step(self, action, render=False):
        if render:
            self.env.render()
        nextState, reward, absorbing, info = self.env.step(action)
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
            fqi (object): an object containing the trained regressor
            expReplay (bool): flag indicating whether to do experience replay
            render (bool): flag indicating whether to render visualize behavior of the agent
        Returns:
            a numpy array containing results of the episode
        
        """
        self._reset()

        return self._runEpisode(fqi, expReplay, render)