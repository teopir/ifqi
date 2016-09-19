from __future__ import print_function
import numpy as np


def discretizeActions(actionMatrix, actionNumber):
    """
    Uniform discretization of the actions in the range observed in the data.

    Example:
        Let the input_matrix be a N x 2 matrix and action_number be [2,3].
        This means that each action has dimension 2 (2 action variables) and we
        want to obtain a discretization with 2 elements along the first
        dimension and 3 elements along the second one.
        We first compute the range of each dimension from the dataset and we
        build a grid 2 x 3 by discretizing the ranges.
        The output is a 6 x 2 matrix.

    Args:
        actionMatrix: array-like
            Samples of actions
        actionNumber: int, array-like
            Number of discrete actions for each action variable

    Returns:
        actions: np.array
            Array of discrete actions (e.g., grid in a 2D problem)
    """
    if len(actionMatrix.shape) == 1:
        actionMatrix = actionMatrix.reshape(-1, 1)
    actionDim = actionMatrix.shape[1]
    # select unique actions
    if isinstance(actionNumber, int):
        actionNumber = [actionNumber] * actionDim

    ubound = np.amax(actionMatrix, axis=0)
    lbound = np.amin(actionMatrix, axis=0)
    if actionDim == 1:
        actions = np.linspace(lbound, ubound, actionNumber[0]).reshape(-1, 1)
    else:
        print("not implemented in the general case (actionDim > 1")
        exit(9)
