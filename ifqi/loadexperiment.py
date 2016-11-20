import numpy as np
import json
import warnings
from gym import spaces
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

import envs
from models.mlp import MLP
from models import ensemble
from models.actionregressor import ActionRegressor
from ifqi.fqi.FQI import FQI


def get_MDP(env):
    """
    This function loads the mdp required in the configuration file.
    Args:
        env (string): the name of the environment.
    Returns:
        The required mdp.

    """
    if env == 'CarOnHill':
        return envs.CarOnHill()
    elif env == 'SwingUpPendulum':
        return envs.InvPendulum()
    elif env == 'Acrobot':
        return envs.Acrobot()
    elif env == "BicycleBalancing":
        return envs.Bicycle(navigate=False)
    elif env == "BicycleNavigate":
        return envs.Bicycle(navigate=True)
    elif env == "SwingPendulum":
        return envs.SwingPendulum()
    elif env == "CartPole":
        return envs.CartPole()
    elif env == "LQG1D":
        return envs.LQG1D()
    else:
        raise ValueError('unknown mdp requested.')


def get_model(name):
    """
    This function loads the model required in the configuration file.
    Returns:
        the required model.
    """
    if name == 'ExtraTree':
        model = ExtraTreesRegressor
    elif name == 'ExtraTreeEnsemble':
        model = ExtraTreeEnsemble
    elif name == 'MLP':
        model = MLP
    elif name == 'MLPEnsemble':
        model = MLPEnsemble
    elif name == 'Linear':
        model = LinearRegression
    elif name == 'LinearEnsemble':
        model = LinearEnsemble
    else:
        raise ValueError('unknown estimator requested.')

    return model
