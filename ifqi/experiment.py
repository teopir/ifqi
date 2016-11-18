import json

from gym import spaces
from sklearn.ensemble import ExtraTreesRegressor
from models.mlp import MLP
from sklearn.linear_model import LinearRegression
from models.ensemble import ExtraTreesEnsemble, MLPEnsemble#, LinearEnsemble
from models.actionregressor import ActionRegressor
from envs.carOnHill import CarOnHill
from envs.invertedPendulum import InvPendulum
from envs.acrobot import Acrobot
from envs.bicycle import Bicycle
from envs.swingPendulum import SwingPendulum
from envs.cartPole import CartPole
from envs.lqg1d import LQG1D
import ifqi.envs as envs
import envs.utils as spaceInfo
from ifqi.fqi.FQI import FQI
import warnings

import numpy as np


class Experiment(object):
    """
    This class has the purpose to load the configuration
    file of the experiment and return the required model
    and mdp.

    """
    def __init__(self, configFile=None):
        """
        Constructor.
        Args:
            config_file (str): the name of the configuration file.

        """
        if configFile is not None:
            with open(configFile) as f:
                self.config = json.load(f)

            self.mdp = self.getMDP()
        else:
            self.config = dict()

    def getModelName(self, nRegressor):
        modelConfig = self.config['regressors'][nRegressor]
        return modelConfig['modelName']

    def getFitParams(self,nRegressor):
        return self.config['regressors'][nRegressor]["supervisedAlgorithm"]

    def getActions(self):
        return self.config['mdp']['discreteActions']

    def getMDP(self, seed=None):
        """
        This function loads the mdp required in the configuration file.
        Returns:
            the required mdp.

        """
        if self.config['mdp']['mdpName'] == 'CarOnHill':
            return CarOnHill()
        elif self.config['mdp']['mdpName'] == 'SwingUpPendulum':
            return InvPendulum()
        elif self.config['mdp']['mdpName'] == 'Acrobot':
            return Acrobot()
        elif self.config["mdp"]["mdpName"] == "BicycleBalancing":
            return Bicycle(navigate=False)
        elif self.config["mdp"]["mdpName"] == "BicycleNavigate":
            return Bicycle(navigate=True)
        elif self.config["mdp"]["mdpName"] == "SwingPendulum":
            return SwingPendulum()
        elif self.config["mdp"]["mdpName"] == "CartPole":
            return CartPole()
        elif self.config["mdp"]["mdpName"] == "CartPoleDisc":
            return CartPole(discreteRew=True)
        elif self.config["mdp"]["mdpName"] == "LQG1D":
            return LQG1D()
        elif self.config["mdp"]["mdpName"] == "LQG1DDisc":
            mdp = LQG1D()
            mdp.discreteReward = True
            return mdp
        else:
            raise ValueError('Unknown mdp type.')

    def _getModel(self, index):
        """
        This function loads the model required in the configuration file.
        Returns:
            the required model.

        """

        stateDim, actionDim = envs.get_space_info(self.mdp)
        modelConfig = self.config['regressors'][index]

        fitActions = False
        if 'fitActions' in modelConfig:
            fitActions = modelConfig['fitActions']

        if modelConfig['modelName'] == 'ExtraTree':
            model = ExtraTreesRegressor
            params = {'n_estimators': modelConfig['nEstimators'],
                      'criterion': self.config["regressors"][index]['supervisedAlgorithm']
                                              ['criterion'],
                      'min_samples_split': modelConfig['minSamplesSplit'],
                      'min_samples_leaf': modelConfig['minSamplesLeaf']}
        elif modelConfig['modelName'] == 'ExtraTreeEnsemble':
            model = ExtraTreesEnsemble
            params = {'nEstimators': modelConfig['nEstimators'],
                      'criterion': self.config["regressors"][index]['supervisedAlgorithm']
                                              ['criterion'],
                      'minSamplesSplit': modelConfig['minSamplesSplit'],
                      'minSamplesLeaf': modelConfig['minSamplesLeaf']}
        elif modelConfig['modelName'] == 'MLP':
            model = MLP
            params = {'n_input': stateDim,
                      'n_output': 1,
                      'hidden_neurons': modelConfig['hidden_neurons'],
                      'optimizer': modelConfig['optimizer'],
                      'activation': modelConfig['activation']}
            if fitActions:
                params["n_input"] = stateDim + actionDim
        elif modelConfig['modelName'] == 'MLPEnsemble':
            model = MLPEnsemble
            params = {'n_input': stateDim,
                      'n_output': 1,
                      'hidden_neurons': modelConfig['hidden_neurons'],
                      'optimizer': modelConfig['optimizer'],
                      'activation': modelConfig['activation']}
            if fitActions:
                params["n_input"] = stateDim + actionDim
        elif modelConfig['modelName'] == 'Linear':
            model = LinearRegression
            params = {}
        elif modelConfig['modelName'] == 'LinearEnsemble':
            #model = LinearEnsemble
            params = {}
        else:
            raise ValueError('Unknown estimator type.')



        if fitActions:
            return model(**params)
        else:
            if isinstance(self.mdp.action_space, spaces.Box):
                warnings.warn("Action Regressor cannot be used for continuous "
                              "action environment. Single regressor will be "
                              "used.")
                return model(**params)
            return ActionRegressor(model,
                                   self.mdp.action_space.values, decimals=5,
                                   **params)

    def getFQI(self, regressorIndex):
        regressor = self._getModel(regressorIndex)
        gamma = self.config['rlAlgorithm']['gamma']
        horizon = self.config['rlAlgorithm']['horizon']
        verbose = self.config['rlAlgorithm']['verbosity']
        scaled = self.config['rlAlgorithm']['scaled']
        optimized=False
        if "optimized" in self.config["rlAlgorithm"]:
            optimized = self.config['rlAlgorithm']['optimized']
        #TODO: fix
        if 'features' in self.config['regressors'][regressorIndex]:
            features = self.config['regressors'][regressorIndex]['features']
        else:
            features = None
            
        state_dim = self.mdp.observation_space.shape[0]

        if(isinstance(self.mdp.action_space, spaces.Box)):
            discreteActions = self.getActions()
        else:
            discreteActions = self.mdp.action_space.values

        fqi = FQI(estimator=regressor,
          state_dim=state_dim,
          #TODO: Fix action dimension
          action_dim=1,
          discrete_actions=discreteActions,
          gamma=gamma,
          horizon=horizon,
          verbose=verbose,
          features=features,
          scaled=scaled)
          
        return fqi 