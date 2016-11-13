import numpy as np
import gym


class DiscreteValued(gym.Space):
    def __init__(self, values, decimals=6):
        """
        Each row represents an action
        """
        assert len(values) > 0, 'list cannot be empty'
        self.decimals = decimals
        self.values = np.around(values, decimals=decimals)

    def sample(self):
        idx = np.random.randint(self.values.shape[0])
        return np.array([self.values[idx]])

    def contains(self, x):
        x = np.asscalar(np.around(x, decimals=self.decimals))
        idx = self.values[np.where(self.values == x)]
        if len(idx) == 1:
            return True
        return False

    def get(self, idx=None):
        if idx:
            return self.values
        d = self.values.shape[0]
        if idx >= 0 and idx < d:
            return self.values[idx]

    @property
    def value_dim(self):
        """
        Return the dimension of the values
        """
        if len(self.values.shape) == 1:
            return 1
        else:
            return self.values.shape[1]

    @property
    def values_count(self):
        """
        Return the number of discrete values
        """
        return self.values.shape[0]

    def __repr__(self):
        return "DiscreteValued({})".format(self.values.tolist)

    def __eq__(self, other):
        return self.n == other.n
