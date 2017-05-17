from __future__ import print_function
import numpy as np


def check_dataset(data, state_dim, action_dim, reward_dim):
    n_columns = 2 * state_dim + action_dim + reward_dim + 2
    assert data.shape[1] == n_columns, \
        '{} != {}'.format(data.shape[1], n_columns)

    state_mask = np.zeros(n_columns, dtype=bool)
    state_mask[:state_dim] = True
    nextstate_mask = np.zeros(n_columns, dtype=bool)
    nextstate_idx = state_dim + action_dim + reward_dim
    nextstate_mask[nextstate_idx:nextstate_idx + state_dim] = True

    for i in range(data.shape[0] - 1):
        snext = data[i, nextstate_mask]
        end_ep = data[i, -1]
        s = data[i + 1, state_mask]
        assert np.allclose(s, snext) or end_ep == 1, \
            '{} != {}'.format(s, snext)


def split_dataset(dataset, state_dim, action_dim, reward_dim, last=None):
    nextstate_idx = state_dim + action_dim + reward_dim
    reward_idx = action_dim + state_dim
    state = dataset[:last, 0:state_dim]
    actions = dataset[:last, state_dim:reward_idx]
    reward = dataset[:last, reward_idx]
    next_states = dataset[:last, nextstate_idx:nextstate_idx + state_dim]
    absorbing = dataset[:last, -2]
    return state, actions, reward, next_states, absorbing


def split_data_for_fqi(dataset, state_dim, action_dim, reward_dim, last=None):
    reward_idx = state_dim + action_dim
    sast = np.append(dataset[:last, :reward_idx],
                     dataset[:last, reward_idx + reward_dim:-1],
                     axis=1)
    r = dataset[:last, reward_idx]
    return sast, r


def filter_state_with_RFS(state, selected_states):
    indexes = [int(s.lstrip('S')) for s in selected_states if s.startswith('S')]
    return np.array(state)[indexes]