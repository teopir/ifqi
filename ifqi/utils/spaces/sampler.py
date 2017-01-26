import numpy as np
import gym.spaces as sp


def space_sampler(space):
    if isinstance(space, sp.Box):
        return lambda: np.random.uniform(low=space.low, high=space.high, size=space.low.shape)
    elif isinstance(space, sp.Discrete):
        return lambda: np.random.randint(space.n)

