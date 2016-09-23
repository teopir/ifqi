import json
from sklearn.ensemble import ExtraTreesRegressor
from models.mlp import MLP
from sklearn.linear_model import LinearRegression
from models.ensemble import ExtraTreeEnsemble, MLPEnsemble, LinearEnsemble
from models.actionRegressor import ActionRegressor
from envs.carOnHill import CarOnHill
from envs.invertedPendulum import InvPendulum
from envs.acrobot import Acrobot
from envs.bicycle import Bicycle
from envs.swingPendulum import SwingPendulum
from envs.cartPole import CartPole
from envs.lqg1d import LQG1D
from envs.lqg1dDiscrete import LQG1DDiscrete
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

    def loadModel(self):
        self.model = self._getModel()

    def getMDP(self):
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
            downerAct = np.linspace(-10,-3,8)
            upperAct = np.linspace(3,10,8)
            middle = np.linspace(-2,2,13)
            actions = downerAct.tolist() + middle.tolist() + upperAct.tolist()        
            return LQG1DDiscrete(actions)
        else:
            raise ValueError('Unknown mdp type.')

    def _getModel(self):
        """
        This function loads the model required in the configuration file.
        Returns:
            the required model.

        """
        modelConfig = self.config['model']
        if modelConfig['modelName'] == 'ExtraTree':
            model = ExtraTreesRegressor
            params = {'n_estimators': modelConfig['nEstimators'],
                      'criterion': self.config['supervisedAlgorithm']
                                              ['criterion'],
                      'min_samples_split': modelConfig['minSamplesSplit'],
                      'min_samples_leaf': modelConfig['minSamplesLeaf']}
        elif modelConfig['modelName'] == 'ExtraTreeEnsemble':
            model = ExtraTreeEnsemble
            params = {'nEstimators': modelConfig['nEstimators'],
                      'criterion': self.config['supervisedAlgorithm']
                                              ['criterion'],
                      'minSamplesSplit': modelConfig['minSamplesSplit'],
                      'minSamplesLeaf': modelConfig['minSamplesLeaf']}
        elif modelConfig['modelName'] == 'MLP':
            model = MLP
            params = {'nInput': self.mdp.stateDim,
                      'nOutput': 1,
                      'hiddenNeurons': modelConfig['nHiddenNeurons'],
                      'nLayers': modelConfig['nLayers'],
                      'optimizer': modelConfig['optimizer'],
                      'activation': modelConfig['activation']}
        elif modelConfig['modelName'] == 'MLPEnsemble':
            model = MLPEnsemble
            params = {'nInput': self.mdp.stateDim,
                      'nOutput': 1,
                      'hiddenNeurons': modelConfig['nHiddenNeurons'],
                      'nLayers': modelConfig['nLayers'],
                      'optimizer': modelConfig['optimizer'],
                      'activation': modelConfig['activation']}
        elif modelConfig['modelName'] == 'Linear':
            model = LinearRegression
            params = {}
        elif modelConfig['modelName'] == 'LinearEnsemble':
            model = LinearEnsemble
            params = {}
        else:
            raise ValueError('Unknown estimator type.')
        
        actionRegressorWrap = True
        if "actionRegressorWrap" in modelConfig:
            actionRegressorWrap = modelConfig["actionRegressorWrap"]
        
        if actionRegressorWrap:
            if self.mdp.nActions < 1:
                warnings.warn("Action Regressor shouldn't be used for continuos action environment")
            return ActionRegressor(model, self.mdp.nActions, **params)
        else:
            return model(**params)

            
