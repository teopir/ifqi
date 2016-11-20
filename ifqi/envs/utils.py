from __future__ import print_function

import gym.spaces as spaces
from ifqi.utils.spaces.discretevalued import DiscreteValued


def get_space_info(env):
    state_space = env.observation_space
    if isinstance(state_space, spaces.Box):
        state_dim = state_space.shape[0]
    elif isinstance(state_space, spaces.Discrete):
        state_dim = 1
    elif isinstance(state_space, DiscreteValued):
        state_dim = state_space.value_dim
    else:
        raise NotImplementedError

    action_space = env.action_space
    if isinstance(action_space, spaces.Box):
        action_dim = action_space.shape[0]
    elif isinstance(action_space, spaces.Discrete):
        action_dim = 1
    elif isinstance(action_space, DiscreteValued):
        action_dim = action_space.value_dim
    else:
        raise NotImplementedError

    reward_dim = 1

    return state_dim, action_dim, reward_dim
