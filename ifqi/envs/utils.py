from __future__ import print_function

import gym.spaces as spaces
from ifqi.utils.spaces.discretevalued import  DiscreteValued

def getSpaceInfo(env):
    
    state_space = env.observation_space
    if isinstance(state_space, 'spaces.Box'):
        stateDim = state_space.shape[0]
    elif isinstance(state_space, 'spaces.Discrete'):
        stateDim = 1
    elif isinstance(state_space, 'DiscreteValued'):
        stateDim = state_space.action_dim()
    else:
        raise NotImplementedError
    
    action_space = env.action_space
    if isinstance(action_space, 'spaces.Box'):
        actionDim = action_space.shape[0]
    elif isinstance(action_space, 'spaces.Discrete'):
        actionDim = 1
    elif isinstance(action_space, 'DiscreteValued'):
        actionDim = action_space.action_dim()
    else:
        raise NotImplementedError
    
    return stateDim, actionDim