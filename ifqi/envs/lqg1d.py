"""classic Linear Quadratic Gaussian Regulator task"""
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from environment import Environment

"""
Linear quadratic gaussian regulator task.

References
----------
  - Simone Parisi, Matteo Pirotta, Nicola Smacchia,
    Luca Bascetta, Marcello Restelli,
    Policy gradient approaches for multi-objective sequential decision making
    2014 International Joint Conference on Neural Networks (IJCNN)
  - J.  Peters  and  S.  Schaal,
    Reinforcement  learning of motor  skills  with  policy  gradients,
    Neural  Networks, vol. 21, no. 4, pp. 682-697, 2008.

"""

# #classic_control
# register(
#     id='LQG1D-v0',
#     entry_point='gym.envs.classic_control:LQG1DEnv',
#     timestep_limit=300,
# )


class LQG1D(gym.Env, Environment):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.stateDim = 1
        self.actionDim = 1
        self.max_pos = 100.0
        self.max_action = 50.0
        self.sigma_noise = 2.0
        self.A = 1
        self.b = 1
        self.Q = 1
        self.R = 1
        self.viewer = None

        high = np.array([self.max_pos])
        self.action_space = spaces.Box(low=-self.max_action,
                                       high=self.max_action,
                                       shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action, render=False):
        u = np.clip(action, -self.max_action, self.max_action)
        noise = np.random.randn() * self.sigma_noise
        xn = np.dot(self.A, self.state) + np.dot(self.b, u) + noise
        cost = np.dot(self.state,
                      np.dot(self.Q, self.state)) + np.dot(u, np.dot(self.R,
                                                                     u))
        # print(self.state, u, noise, xn, cost)

        self.state = xn
        # return xn, -cost, False, {}
        return -cost

    def _reset(self, state=None):
        self.state = np.array([self.np_random.uniform(low=-10, high=-10)])
        return np.array(self.state)

    def _getState(self):
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.max_pos * 2
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

    def evaluate(self, fqi, expReplay=False, render=False):
        """
        This function evaluates the regressor in the provided object
        parameter.
        This way of evaluation is just one of many possible ones.
        Params:
            fqi (object): an object containing the trained regressor
            expReplay (bool): flag indicating whether to do experience
                              replay
            render (bool): flag indicating whether to render visualize
                           behavior of the agent
        Returns:
            a numpy array containing results of the episode

        """
        self._reset()
        return self.runEpisode(fqi, expReplay, render)
