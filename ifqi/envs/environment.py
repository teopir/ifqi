import gym
from gym.utils import seeding


class Environment(gym.Env):
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
