import gym
import math

from gymEnv import GymEnv


class CartPole(GymEnv):
    def __init__(self, discreteRew=False):
        self.env = gym.make('CartPole-v0')
        self.env.reset()
        self.stateDim = 4
        self.actionDim = 1
        self.nStates = 0
        self.nActions = 2
        self.horizon = 400
        self.gamma = 0.99
        self._absorbing = False
        self._atGoal = False
        self._nextState = self.env._reset().tolist()
        self._discreteRew=discreteRew
        self._count=0

    def _reset(self):
        self._absorbing = False
        self._atGoal = False
        self.env.reset()
        self._nextState = self.env._reset().tolist()
        self._count=0

    def _step(self, action, render=False):
        self._count += 1
        if render:
            self.env.render()
        if isinstance(action, int):
            nextState, reward, absorbing, info = self.env.step(action)
        else:
            nextState, reward, absorbing, info = self.env.step(int(action[0]))

        x = nextState[0]
        theta = nextState[2]
        
        if not self._discreteRew:
            theta_rew = (math.cos(theta) -
                         math.cos(self.env.theta_threshold_radians)) / (
                         1. - math.cos(self.env.theta_threshold_radians))
            x_rew = - abs(x / (self.env.x_threshold))
            reward = theta_rew + x_rew
        
        self._nextState = nextState
        self._absorbing = absorbing
        
        if(self._count>=self.horizon):
            self._atGoal=True
            
        return reward

    def _getState(self):
        return self._nextState

    def evaluate(self, fqi, expReplay=False, render=False, n_episodes=1):
        """
        This function evaluates the regressor in the provided object parameter.
        This way of evaluation is just one of many possible ones.
        Params:
            fqi (object): an object containing the trained regressor
            expReplay (bool): flag indicating whether to do experience replay
            render (bool): flag indicating whether to render visualize behavior
                           of the agent
        Returns:
            a numpy array containing results of the episode

        """
        self._reset()
        if not expReplay:
            J=0.
            step=0
            nGoal=0
            for i in range(n_episodes):
                self._reset()
                j, s, t = self.runEpisode(fqi, expReplay, render)
                J+=j
                step+=s
                nGoal+=t
            return J/(n_episodes +0.),step/(n_episodes +0.),nGoal/(n_episodes +0.)
        else:
            return self.runEpisode(fqi, expReplay, render)
