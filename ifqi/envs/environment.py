import gym
from gym.utils import seeding
from .. import evaluation as evaluation


class Environment(gym.Env):
    def __init__(self):
        self.gamma = 1
        self.horizon = None

    def set_seed(self, seed=None):
        self._seed(seed=seed)
