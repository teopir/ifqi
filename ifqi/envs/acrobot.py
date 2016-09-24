# import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint

from environment import Environment

"""
The Acrobot environment as presented in:

- Ernst, Damien, Pierre Geurts, and Louis Wehenkel.
  "Tree-based batch mode reinforcement learning."
  Journal of Machine Learning Research 6.Apr (2005): 503-556.

This problem has continuous state (4-dim) and discrete actions (by default).
"""


class Acrobot(Environment):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self):
        self.horizon = 100
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
        self.max_action = 5

        # gym attributes
        self.viewer = None
        high = np.array([np.inf, np.inf, np.inf, np.inf])
        low = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
        self.observation_space = spaces.Box(low=low, high=high)
        higha = np.array([self.max_action])
        self.action_space = spaces.Box(low=-higha, high=higha)

        # initialize state
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, u, render=False):
        u = np.clip(u, -self.max_action, self.max_action)

        theta1, theta2 = self.state[0], self.state[1]
        dTheta1, dTheta2 = self.state[2], self.state[3]
        newState = odeint(self._dpds,
                          np.array([theta1, theta2, dTheta1, dTheta2, u]),
                          [0, self._dt],
                          rtol=1e-5, atol=1e-5, mxstep=2000)

        x = np.array(newState[-1][0:4])

        k = round((x[0] - np.pi) / (2 * np.pi))
        o = np.array([2 * k * np.pi + np.pi, 0., 0., 0.])
        d = np.linalg.norm(x - o)

        x[0] = self._wrap2pi(x[0])
        x[1] = self._wrap2pi(x[1])

        self.state = np.array(x)

        reward = 0
        if d < 1:
            self._absorbing = True
            self._atGoal = True
            reward = 1 - d
        return x, reward, self._absorbing, {}

    def _reset(self, state=None):
        self._absorbing = False
        self._atGoal = False
        if state is None:
            theta1 = self._wrap2pi(self.np_random.uniform(low=-np.pi + 1, high=np.pi - 1))
            theta2 = dTheta1 = dTheta2 = 0
        else:
            theta1 = self._wrap2pi(state[0])
            theta2 = self._wrap2pi(state[1])
            dTheta1 = state[2]
            dTheta2 = state[3]
        self.state = np.array([theta1, theta2, dTheta1, dTheta2])
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

        p1 = [-self._L1 *
              np.cos(s[0]), self._L1 * np.sin(s[0])]

        p2 = [p1[0] - self._L2 * np.cos(s[0] + s[1]),
              p1[1] + self._L2 * np.sin(s[0] + s[1])]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - np.pi / 2, s[0] + s[1] - np.pi / 2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x, y), th) in zip(xys, thetas):
            l, r, t, b = 0, 1, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, .8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _getState(self):
        return np.array(self.state)

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

        return diffTheta1, diffTheta2, diffDiffTheta1, diffDiffTheta2, 0.

    def _wrap2pi(self, value):
        tmp = value - -np.pi
        width = 2 * np.pi
        tmp -= width * np.floor(tmp / width)

        return tmp + -np.pi

    def evaluate(self, policy, nbEpisodes=1, metric='discounted', render=False):
        """
        This function evaluates the regressor in the provided object parameter.
        This way of evaluation is just one of many possible ones.
        Params:
            policy (object): a policy object (method drawAction is expected)
            nbEpisodes (int): the number of episodes to execute
            metric (string): the evaluation metric ['discounted', 'average']
            render (bool): flag indicating whether to visualize the behavior of
                            the policy
        Returns:
            a numpy array containing the average score obtained starting from
            41 different states

        """
        states = np.linspace(-2, 2, 41)
        nstates = states.size

        values = np.zeros(nstates)
        counter = 0
        for theta1 in states:
            self._reset([theta1, 0., 0., 0.])
            values[counter] = super(Acrobot, self).evaluate(policy, nbEpisodes, metric, render)[0]
            counter += 1

        return values.mean(), 2 * values.std() / np.sqrt(nstates)
