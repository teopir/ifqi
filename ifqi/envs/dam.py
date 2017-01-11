"""classic Linear Quadratic Gaussian Regulator task"""
from numbers import Number

import gym
import numpy as np
from gym import spaces
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

# classic_control
from gym.envs.registration import register

register(
    id='Dam-v0',
    entry_point='ifqi.envs.dam:Dam',
    timestep_limit=300,
)


class Dam(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, dreward=2, penalize=False):
        assert 0 < dreward < 5
        self.reward_dim = dreward
        self.horizon = 100
        self.gamma = 1.0
        self.isAveraged = True
        self.penalize = penalize  # True to penalize the policy when it violates the problem's constraints

        self.S = 1.0  # Reservoir surface
        self.W_IRR = 50.0  # Water demand
        self.H_FLO_U = 50.0  # Flooding threshold (upstream, i.e., height of the dam)
        self.S_MIN_REL = 100.0  # Release threshold (i.e., max capacity)
        self.DAM_INFLOW_MEAN = 40  # Random inflow (e.g. rain)
        self.DAM_INFLOW_STD = 10
        self.Q_MEF = 0.0
        self.GAMMA_H2O = 1000.0  # Water density
        self.W_HYD = 4.36  # Hydroelectric demand
        self.Q_FLO_D = 30.0  # Flooding threshold (downstream, i.e., releasing too much water)
        self.ETA = 1.0  # Turbine efficiency
        self.G = 9.81  # Gravity

        # gym attributes
        self.viewer = None
        self.action_space = spaces.Box(low=0.0,
                                       high=np.inf,
                                       shape=(1,))
        self.observation_space = spaces.Box(low=0.0,
                                            high=np.inf,
                                            shape=(1,))

        # initialize state
        self.seed()
        self.reset()

    def step(self, action, render=False):
        reward = np.zeros(4)

        # Bound the action
        actionLB = max(self.state - self.S_MIN_REL, 0.0)
        actionUB = self.state

        # Penalty proportional to the violation
        bounded_action = min(max(action, actionLB), actionUB)
        penalty = -self.penalize * abs(bounded_action - action)

        # Transition dynamic
        action = bounded_action
        dam_inflow = self.DAM_INFLOW_MEAN + np.random.randn() * self.DAM_INFLOW_STD
        nextstate = max(self.state + dam_inflow - action, 0)  # There is a very small chance that dam_inflow < 0

        # Cost due to the excess level w.r.t. a flooding threshold (upstream)
        reward[0] = -max(nextstate / self.S - self.H_FLO_U, 0.0) + penalty

        # Deficit in the water supply w.r.t. the water demand
        reward[1] = -max(self.W_IRR - action, 0.0) + penalty

        q = max(action - self.Q_MEF, 0.0)
        p_hyd = self.ETA * self.G * self.GAMMA_H2O * nextstate / self.S * q / (3.6e6)

        # Deficit in the hydroelectric supply w.r.t. the hydroelectric demand
        reward[2] = -max(self.W_HYD - p_hyd, 0.0) + penalty

        # Cost due to the excess level w.r.t. a flooding threshold (downstream)
        reward[3] = -max(action - self.Q_FLO_D, 0.0) + penalty

        reward = reward[0:self.reward_dim]
        absorb = False
        self.state = nextstate

        return self.get_state(), reward, False, {}

    def reset(self, state=None):
        if state is None:
            if self.penalize:
                self.state = prng.np_random.uniform(0.0, 160.0)
            else:
                s_init = np.array([9.6855361e+01, 5.8046026e+01,
                                   1.1615767e+02, 2.0164311e+01,
                                   7.9191000e+01, 1.4013098e+02,
                                   1.3101816e+02, 4.4351321e+01,
                                   1.3185943e+01, 7.3508622e+01,
                                   ])
                idx = prng.np_random.randint(low=0, high=s_init.size())
                self.state = np.asscalar(s_init[idx])
        else:
            assert np.isscalar(state) and state > 0.
            self.state = np.asscalar(state)

        return self.get_state()

    def get_state(self):
        return np.array(self.state)

    def true_frontier(self):
        fname = 'dam_front{}d.dat'.format(self.reward_dim)
        front = np.loadtxt(fname)
        fname = 'dam_w{}d.dat'.format(self.reward_dim)
        weights = np.loadtxt(fname)
        return front, weights
