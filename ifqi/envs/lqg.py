"""classic Linear Quadratic Gaussian Regulator task"""
# import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from ifqi.envs.environment import Environment

from gym.spaces import prng
"""
Linear quadratic gaussian regulator task.

References
----------
  - Simone Parisi, Matteo Pirotta, Nicola Smacchia,
    Luca Bascetta, Marcello Restelli,
    Policy gradient approaches for multi-objective sequential decision making
    2014 International Joint Conference on Neural Networks (IJCNN)
  - Jan  Peters  and  Stefan  Schaal,
    Reinforcement  learning of motor  skills  with  policy  gradients,
    Neural  Networks, vol. 21, no. 4, pp. 682-697, 2008.

"""

'''
#classic_control
from gym.envs.registration import register
register(
    id='LQG1D-v0',
    entry_point='ifqi.envs.lqg1d:LQG1D',
    timestep_limit=300,
)
'''

class LQG(Environment):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,
                 dimensions=2):
        self.horizon = 100
        self.gamma = 0.99
        self.dimensions = dimensions

        self.max_pos = 10.0
        self.max_action = 8.0
        self.sigma_noise = 0.01 * np.eye(dimensions)
        #self.A = np.eye(dimensions)
        #self.B = np.eye(dimensions)

        self.A = np.array([[1., 0.1],
                           [0.1, 1.]])
        self.B = np.array([[1., 0.1],
                           [0.1, 1.]])

        self.Q = 0.9 *  np.eye(dimensions)
        self.R = 0.9 *  np.eye(dimensions)

        # gym attributes
        self.viewer = None
        self.action_space = spaces.Box(low=-self.max_action,
                                       high=self.max_action,
                                       shape=(dimensions,))
        self.observation_space = spaces.Box(low=-self.max_pos,
                                            high=self.max_pos,
                                            shape=(dimensions,))

        # initialize state
        self.seed()
        self.reset()

    def get_cost(self, x, u):
        return np.dot(x, np.dot(self.Q, x)) + np.dot(u, np.dot(self.R, u))

    def step(self, action, render=False):
        u = np.clip(action, -self.max_action, self.max_action)
        noise = self.np_random.multivariate_normal(np.zeros(self.dimensions), self.sigma_noise)
        xn = np.dot(self.A, self.state) + np.dot(self.B, u) + noise
        cost = self.get_cost(self.state, u)

        self.state = np.array(xn.ravel())
        return self.get_state(), -np.asscalar(cost), False, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None):

        if state is None:
            self.state = np.array(prng.np_random.uniform(-self.max_pos,
                                                          self.max_pos,
                                                          self.dimensions))
        else:
            self.state = np.array(state)

        return self.get_state()

    def get_state(self):
        return self.state
