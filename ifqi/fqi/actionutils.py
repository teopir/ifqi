from __future__ import print_function
import numpy as np


def discretize_actions(action_matrix, action_number):
    """
    Uniform discretization of the actions in the range observed in the data.

    Example:
        Let the input_matrix be a N x 2 matrix and action_number be [2,3].
        This means that each action has dimension 2 (2 action variables) and we
        want to obtain a discretization with 2 elements along the first dimension and
        3 elements along the second one.
        We first compute the range of each dimension from the dataset and we build a grid
        2 x 3 by discretizing the ranges.
        The output is a 6 x 2 matrix.

    Parameters
    ----------
    action_matrix: array-like
        Samples of actions
    action_number: int, array-like
        Number of discrete actions for each action variable
    Returns
    -------
    actions: np.array
        Array of discrete actions (e.g., grid in a 2D problem)
    """
    if len(action_matrix.shape) == 1:
        action_matrix = action_matrix.reshape(-1,1)
    action_dim = action_matrix.shape[1]
    # select unique actions
    if isinstance(action_number, int):
        action_number = [action_number] * action_dim

    ubound = np.amax(action_matrix, axis=0)
    lbound = np.amin(action_matrix, axis=0)
    if action_dim == 1:
        actions = np.linspace(lbound, ubound, action_number[0]).reshape(-1, 1)
    else:
        print("not implemented in the general case (action_dim > 1")
        exit(9)
