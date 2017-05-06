"""classic Linear Quadratic Gaussian Regulator task"""
# import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from environment import Environment

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

class LQG1D(Environment):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, discrete_reward=False):
        self.horizon = 100
        self.gamma = 0.99

        self.discrete_reward = discrete_reward
        self.max_pos = 10.0
        self.max_action = 8.0
        self.sigma_noise = 0.1
        self.A = np.array([1]).reshape((1, 1))
        self.B = np.array([1]).reshape((1, 1))
        self.Q = np.array([0.9]).reshape((1, 1))
        self.R = np.array([0.9]).reshape((1, 1))

        # gym attributes
        self.viewer = None
        high = np.array([self.max_pos])
        self.action_space = spaces.Box(low=-self.max_action,
                                       high=self.max_action,
                                       shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        # initialize state
        self.seed()
        self.reset()

    def get_cost(self, x, u):
        return np.dot(x, np.dot(self.Q, x)) + np.dot(u, np.dot(self.R, u))

    def step(self, action, render=False):
        u = np.clip(action, -self.max_action, self.max_action)
        noise = self.np_random.randn() * self.sigma_noise
        xn = np.dot(self.A, self.state) + np.dot(self.B, u) + noise
        cost = self.get_cost(self.state, u)

        self.state = np.array(xn.ravel())
        if self.discrete_reward:
            if abs(self.state[0]) <= 2 and abs(u) <= 2:
                return self.get_state(), 0, False, {}
            return self.get_state(), -1, False, {}
        return self.get_state(), -np.asscalar(cost), False, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None):
        '''
        if state is None:
            self.state = np.array([prng.np_random.uniform(low=-self.max_pos,
                                                          high=self.max_pos)])
        else:
            self.state = np.array(state)
        '''
        self.state = np.array([4.])

        return self.get_state()

    def get_state(self):
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = (self.max_pos * 2) * 2
        scale = screen_width / world_width
        bally = 100
        ballradius = 3

        if self.viewer is None:
            clearance = 0  # y-offset
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            mass = rendering.make_circle(ballradius * 2)
            mass.set_color(.8, .3, .3)
            mass.add_attr(rendering.Transform(translation=(0, clearance)))
            self.masstrans = rendering.Transform()
            mass.add_attr(self.masstrans)
            self.viewer.add_geom(mass)
            self.track = rendering.Line((0, bally), (screen_width, bally))
            self.track.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(self.track)
            zero_line = rendering.Line((screen_width / 2, 0),
                                       (screen_width / 2, screen_height))
            zero_line.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(zero_line)

        x = self.state[0]
        ballx = x * scale + screen_width / 2.0
        self.masstrans.set_translation(ballx, bally)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _computeP2(self, K):
        """
        This function computes the Riccati equation associated to the LQG
        problem.
        Args:
            K (matrix): the matrix associated to the linear controller K * x

        Returns:
            P (matrix): the Riccati Matrix

        """
        I = np.eye(self.Q.shape[0], self.Q.shape[1])
        if np.array_equal(self.A, I) and np.array_equal(self.B, I):
            P = (self.Q + np.dot(K.T, np.dot(self.R, K))) / (I - self.gamma *
                                                             (I + 2 * K + K **
                                                              2))
        else:
            tolerance = 0.0001
            converged = False
            P = np.eye(self.Q.shape[0], self.Q.shape[1])
            while not converged:
                Pnew = self.Q + self.gamma * np.dot(self.A.T,
                                                    np.dot(P, self.A)) + \
                       self.gamma * np.dot(K.T, np.dot(self.B.T,
                                                       np.dot(P, self.A))) + \
                       self.gamma * np.dot(self.A.T,
                                           np.dot(P, np.dot(self.B, K))) + \
                       self.gamma * np.dot(K.T,
                                           np.dot(self.B.T,
                                                  np.dot(P, np.dot(self.B,
                                                                   K)))) + \
                       np.dot(K.T, np.dot(self.R, K))
                converged = np.max(np.abs(P - Pnew)) < tolerance
                P = Pnew
        return P

    def computeOptimalK(self):
        """
        This function computes the optimal linear controller associated to the
        LQG problem (u = K * x).

        Returns:
            K (matrix): the optimal controller bv

        """
        P = np.eye(self.Q.shape[0], self.Q.shape[1])
        for i in range(100):
            K = -self.gamma * np.dot(np.linalg.inv(
                self.R + self.gamma * (np.dot(self.B.T, np.dot(P, self.B)))),
                                       np.dot(self.B.T, np.dot(P, self.A)))
            P = self._computeP2(K)
        K = -self.gamma * np.dot(np.linalg.inv(self.R + self.gamma *
                                               (np.dot(self.B.T,
                                                       np.dot(P, self.B)))),
                                 np.dot(self.B.T, np.dot(P, self.A)))
        return K

    def computeJ(self, K, Sigma, n_random_x0=100):
        """
        This function computes the discounted reward associated to the provided
        linear controller (u = Kx + \epsilon, \epsilon \sim N(0,\Sigma)).
        Args:
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
                            the controller action
            n_random_x0: the number of samples to draw in order to average over
                         the initial state

        Returns:
            J (float): The discounted reward

        """
        if isinstance(K, (int, long, float, complex)):
            K = np.array([K]).reshape(1, 1)
        if isinstance(Sigma, (int, long, float, complex)):
            Sigma = np.array([Sigma]).reshape(1, 1)

        P = self._computeP2(K)
        J = 0.0
        for i in range(n_random_x0):
            self._reset()
            x0 = self._getState()
            J -= np.dot(x0.T, np.dot(P, x0)) \
                + (1 / (1 - self.gamma)) * \
                np.trace(np.dot(
                    Sigma, (self.R + self.gamma * np.dot(self.B.T,
                                                         np.dot(P, self.B)))))
        J /= n_random_x0
        return J

    def computeQFunction(self, x, u, K, Sigma, n_random_xn=100):
        """
        This function computes the Q-value of a pair (x,u) given the linear
        controller Kx + epsilon where epsilon \sim N(0, Sigma).
        Args:
            x (int, array): the state
            u (int, array): the action
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
            the controller action
            n_random_xn: the number of samples to draw in order to average over
            the next state

        Returns:
            Qfun (float): The Q-value in the given pair (x,u) under the given
            controller

        """
        if isinstance(x, (int, long, float, complex)):
            x = np.array([x])
        if isinstance(u, (int, long, float, complex)):
            u = np.array([u])
        if isinstance(K, (int, long, float, complex)):
            K = np.array([K]).reshape(1, 1)
        if isinstance(Sigma, (int, long, float, complex)):
            Sigma = np.array([Sigma]).reshape(1, 1)

        P = self._computeP2(K)
        Qfun = 0
        for i in range(n_random_xn):
            noise = np.random.randn() * self.sigma_noise
            action_noise = np.random.multivariate_normal(
                np.zeros(Sigma.shape[0]), Sigma, 1)
            nextstate = np.dot(self.A, x) + np.dot(self.B,
                                                   u + action_noise) + noise
            Qfun -= np.dot(x.T, np.dot(self.Q, x)) + \
                np.dot(u.T, np.dot(self.R, u)) + \
                self.gamma * np.dot(nextstate.T, np.dot(P, nextstate)) + \
                (self.gamma / (1 - self.gamma)) * \
                np.trace(np.dot(Sigma,
                                self.R + self.gamma *
                                np.dot(self.B.T, np.dot(P, self.B))))
        Qfun = np.asscalar(Qfun) / n_random_xn
        return Qfun

    def computeVFunction(self, x, K, Sigma, n_random_xn=100):
        """
        This function computes the Q-value of a pair (x,u) given the linear
        controller Kx + epsilon where epsilon \sim N(0, Sigma).
        Args:
            x (int, array): the state
            u (int, array): the action
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
            the controller action
            n_random_xn: the number of samples to draw in order to average over
            the next state

        Returns:
            Qfun (float): The Q-value in the given pair (x,u) under the given
            controller

        """
        if isinstance(x, (int, long, float, complex)):
            x = np.array([x])
        if isinstance(K, (int, long, float, complex)):
            K = np.array([K]).reshape(1, 1)
        if isinstance(Sigma, (int, long, float, complex)):
            Sigma = np.array([Sigma]).reshape(1, 1)

        P = self._computeP2(K)
        Vfun = 0
        for i in range(n_random_xn):
            u = np.random.randn() * Sigma + K * x
            noise = np.random.randn() * self.sigma_noise
            action_noise = np.random.multivariate_normal(
                np.zeros(Sigma.shape[0]), Sigma, 1)
            nextstate = np.dot(self.A, x) + np.dot(self.B,
                                                   u + action_noise) + noise
            Vfun -= np.dot(x.T, np.dot(self.Q, x)) + \
                    np.dot(u.T, np.dot(self.R, u)) + \
                    self.gamma * np.dot(nextstate.T, np.dot(P, nextstate)) + \
                    (self.gamma / (1 - self.gamma)) * \
                    np.trace(np.dot(Sigma,
                                    self.R + self.gamma *
                                    np.dot(self.B.T, np.dot(P, self.B))))
        Qfun = np.asscalar(Vfun) / n_random_xn
        return Qfun

        # TODO check following code

        # def computeM(self, K):
        #     kb = np.dot(K, self.B.T)
        #     size = self.A.shape[1] ** 2;
        #     AT = self.A.T
        #     return np.eye(size) - self.gamma * (np.kron(AT, AT) - np.kron(AT, kb) - np.kron(kb, AT) + np.kron(kb, kb))
        #
        # def computeL(self, K):
        #     return self.Q + np.dot(K, np.dot(self.R, K.T))
        #
        # def to_vec(self, m):
        #     n_dim = self.A.shape[1]
        #     v = m.reshape(n_dim * n_dim, 1)
        #     return v
        #
        # def to_mat(self, v):
        #     n_dim = self.A.shape[1]
        #     M = v.reshape(n_dim, n_dim)
        #     return M
        #
        # def computeJ(self, k, Sigma, n_random_x0=100):
        #     J = 0
        #     K = k
        #     if len(k.shape) == 1:
        #         K = np.diag(k)
        #     P = self.computeP(K)
        #     for i in range(n_random_x0):
        #         self._reset()
        #         x0 = self.state
        #         v = np.asscalar(x0.T * P * x0 + np.trace(
        #             np.dot(Sigma, (self.R + np.dot(self.gamma, np.dot(self.B.T, np.dot(P, self.B)))))) / (1.0 - self.gamma))
        #         J += -v
        #     J /= n_random_x0
        #
        #     return J
        #
        # def solveRiccati(self, k):
        #     K = k
        #     if len(k.shape) == 1:
        #         K = np.diag(k)
        #     return self.computeP(K)
        #
        # def riccatiRHS(self, k, P, r):
        #     K = k
        #     if len(k.shape) == 1:
        #         K = np.diag(k)
        #     return self.Q + self.gamma * (np.dot(self.A.T, np.dot(self.P, self.A))
        #                                   - np.dot(K, np.dot(self.B.T, np.dot(self.P, self.A)))
        #                                   - np.dot(self.A.T, np.dot(self.P, np.dot(self.B, K.T)))
        #                                   + np.dot(K, np.dot(self.B.T, np.dot(self.P, np.dot(self.B, K.T))))) \
        #            + np.dot(K, np.dot(self.R, K.T))
        #
        # def computeP(self, K):
        #     L = self.computeL(K)
        #     M = self.computeM(K)
        #
        #     vecP = np.linalg.solve(M, self.to_vec(L))
        #
        #     P = self.to_mat(vecP)
        #     return P
