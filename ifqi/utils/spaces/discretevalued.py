import numpy as np

import gym
from gym.spaces import prng


class DiscreteValued(gym.Space):
    def __init__(self, values, decimals=6):
        assert len(values) > 0, 'list cannot be empty'
        self.decimals = decimals
        self.values = np.around(values, decimals=decimals)

    def sample(self):
        idx = prng.np_random.randint(len(self.values))
        return self.values[idx]

    def contains(self, x):
        x = np.asscalar(np.around(x, decimals=self.decimals))
        idx = self.values[np.where(self.values == x)]
        if len(idx) == 1:
            return True
        return False

    def get(self, idx=None):
        if idx:
            return self.values
        d = len(self.values)
        if idx >= 0 and idx < d:
            return self.values[idx]

    @property
    def shape(self):
        return tuple(len(self.values), )

    def __repr__(self):
        return "DiscreteValued({})".format(self.values.tolist)

    def __eq__(self, other):
        return self.n == other.n
