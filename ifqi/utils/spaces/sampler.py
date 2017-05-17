import numpy as np
import gym.spaces as sp
from .discretevalued import DiscreteValued


def multi_discrete_sampler(space):
    random_array = np.random.rand(space.num_discrete_space)
    return [int(x) for x in np.rint(np.multiply((space.high - space.low), random_array) + space.low)]


def space_sampler(space):
    if isinstance(space, sp.Box):
        return lambda: np.random.uniform(low=space.low, high=space.high, size=space.low.shape)
    elif isinstance(space, sp.Discrete):
        return lambda: np.random.randint(space.n)
    elif isinstance(space, DiscreteValued):
        return lambda: space.sample()
    elif isinstance(space, sp.MultiDiscrete):
        return lambda: multi_discrete_sampler(space)
    elif isinstance(space, sp.Tuple):
        return lambda: tuple([space_sampler(lspace) for lspace in space.spaces])
    else:
        raise ValueError("Unknown space")