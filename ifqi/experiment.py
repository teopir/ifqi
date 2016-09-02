import json

from sklearn.ensemble import ExtraTreesRegressor
from ifqi.models.mlp import MLP
from sklearn.linear_model import LinearRegression
from ifqi.models.ensemble import ExtraTreeEnsemble, MLPEnsemble, LinearEnsemble
from ifqi.models.actionRegressor import ActionRegressor
from ifqi.envs.carOnHill import CarOnHill
from ifqi.envs.invertedPendulum import InvPendulum
from ifqi.envs.acrobot import Acrobot
from ifqi.envs.bicycle import Bicycle


class Experiment(object):
    """
    This class has the purpose to load the configuration
    file of the experiment and return the required model
    and mdp.

    """
    def __init__(self, config_file):
        """
        Constructor.
        Args:
            config_file (str): the name of the configuration file.
        
        """
        with open(config_file) as f:
            self.config = json.load(f)
        
        self.mdp = self._getMDP()
        
    def loadModel(self):
        self.model = self._getModel()
    
    def _getModel(self):
        """
        This function loads the model required in the configuration file.
        Returns:
            the required model.
        
        """
        model_config = self.config['model']
        if model_config['model_name'] == 'ExtraTree':
            model = ExtraTreesRegressor
            params = {'n_estimators':model_config['n_estimators'],
                                        'criterion':self.config['supervised_algorithm']['criterion'],
                                        'min_samples_split':model_config['min_samples_split'],
                                        'min_samples_leaf':model_config['min_samples_leaf']}
        elif model_config['model_name'] == 'ExtraTreeEnsemble':
            model = ExtraTreeEnsemble
            params = {'n_estimators':model_config['n_estimators'],
                                      'criterion':self.config['supervised_algorithm']['criterion'],
                                      'min_samples_split':model_config['min_samples_split'],
                                      'min_samples_leaf':model_config['min_samples_leaf']}
        elif model_config['model_name'] == 'MLP':
            model = MLP
            params = {'n_input':self.mdp.state_dim,
                        'n_output':1,
                        'hidden_neurons':model_config['n_hidden_neurons'],
                        'n_layers':model_config['n_layers'],
                        'optimizer':model_config['optimizer'],
                        'activation':model_config['activation']}
        elif model_config['model_name'] == 'MLPEnsemble':
            model = MLPEnsemble
            params = {'n_input':self.mdp.state_dim,
                                'n_output':1,
                                'hidden_neurons':model_config['n_hidden_neurons'],
                                'n_layers':model_config['n_layers'],
                                'optimizer':model_config['optimizer'],
                                'activation':model_config['activation']}
        elif model_config['model_name'] == 'Linear':
            model = LinearRegression
            params = {}
        elif model_config['model_name'] == 'LinearEnsemble':
            model = LinearEnsemble
            params = {}
        else:
            raise ValueError('Unknown estimator type.')

        return ActionRegressor(model, self.mdp.n_actions, **params)
        
    def _getMDP(self):
        """
        This function loads the mdp required in the configuration file.
        Returns:
            the required mdp.
        
        """
        if self.config['mdp']['mdp_name'] == 'CarOnHill':
            return CarOnHill()
        elif self.config['mdp']['mdp_name'] == 'SwingUpPendulum':
            return InvPendulum()
        elif self.config['mdp']['mdp_name'] == 'Acrobot':
            return Acrobot()
        elif self.config["mdp"]["mdp_name"] == "BicycleBalancing":
            return Bicycle(navigate=False)
        elif self.config["mdp"]["mdp_name"] == "BicycleNavigate":
            return Bicycle(navigate=True)
        else:
            raise ValueError('Unknown mdp type.')