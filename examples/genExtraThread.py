import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from ifqi import envs
import ifqi.evaluation.evaluation as evaluate
from ifqi.fqi.FQI import FQI
from ifqi.models.mlp import MLP
from ifqi.models.regressor import Regressor
from gym.spaces import prng
import random

import argparse


parser = argparse.ArgumentParser(
    description='Execution of one experiment thread provided a configuration file and\n\t A regressor (index)\n\t Size of dataset (index)\n\t Dataset (index)')

ranges = {
    "n_estimators": (1, 50),
    "min_sample_split": (0, 1000),
    "min_sample_leaf": (0, 1000),
    "min_weight_fraction_leaf": (0, 1000),
    "bootstrap": (0, 2),
    "max_depth": (0, 1000),
}

parser.add_argument("env_name", type=str, help="Provide the environment")
parser.add_argument("n_estimators", type=int, help="Provide the size of the population")
parser.add_argument("min_sample_split", type=int, help="Provides the number of core to use")
parser.add_argument("min_sample_leaf", type=int, help="Provides the number of core to use")
parser.add_argument("min_weight_fraction_leaf", type=int, help="Provides the number of core to use")
parser.add_argument("bootstrap", type=int, help="Provides the number of core to use")
parser.add_argument("max_depth", type=int, help="Provides the number of core to use")
parser.add_argument("seed", type=int, help="Provides the seed of randomness")



args = parser.parse_args()

env_name = args.env_name
n_estimators = args.n_estimators
min_sample_split = args.min_sample_split #/ 1000.
min_sample_leaf = args.min_sample_leaf #/ 1000.
min_weight_fraction_leaf = args.min_weight_fraction_leaf / 2000.
bootstrap = args.bootstrap==1
max_depth = args.max_depth if args.max_depth > 1 else None
input_scaled = False
output_scaled = False
seed = args.seed



"""Easy function
print -((-nEpochs + 520)**2 + 1) * ((- nNeurons + 21)**2 + 1)
"""

prng.seed(seed)
np.random.seed(seed)
random.seed(seed)

sizeDS = None
if env_name=="SwingPendulum":
    sizeDS = 2000
    mdp = envs.SwingPendulum()
    discrete_actions = [-5.,5.]
elif env_name=="Acrobot":
    sizeDS = 2000
    mdp = envs.Acrobot()
    discrete_actions = mdp.action_space.values
elif env_name=="LQG1D":
    sizeDS = 5
    mdp = envs.LQG1D()
    discrete_actions = [-5.,-2.5,-1.,-0.5, 0., 0.5, 1., 2.5, 5]
elif env_name=="LQG1DD":
    sizeDS = 5
    mdp = envs.LQG1D(discrete_reward=True)
    discrete_actions = [-5.,-2.5,-1.,-0.5, 0., 0.5, 1., 2.5, 5]
elif env_name=="Bicycle":
    sizeDS = 5
    mdp = envs.Bicycle()
    discrete_actions = [-5.,-2.5,-1.,-0.5, 0., 0.5, 1., 2.5, 5]

mdp.seed(0)

state_dim, action_dim = envs.get_space_info(mdp)

regressor_params = {"n_estimators":n_estimators,
                    "max_depth":max_depth,
                    "min_samples_split":min_sample_split,
                    "min_samples_leaf":min_sample_leaf,
                    "min_weight_fraction_leaf":min_weight_fraction_leaf,
                    "bootstrap":bootstrap}

regressor_params["input_scaled"]= input_scaled==1
regressor_params["output_scaled"]= output_scaled==1
regressor = Regressor(regressor_class=ExtraTreesRegressor, **regressor_params)

state_dim, action_dim = envs.get_space_info(mdp)
reward_idx = state_dim + action_dim
dataset = evaluate.collect_episodes(mdp,policy=None,n_episodes=sizeDS)
sast = np.append(dataset[:, :reward_idx], dataset[:, reward_idx + 1:], axis=1)
sastFirst, rFirst = sast, dataset[:, reward_idx]

fqi = FQI(estimator=regressor,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=mdp.gamma,
          horizon=mdp.horizon,
          features=None,
          verbose=True)


fitParams = {
}
fqi.partial_fit(sastFirst[:], rFirst[:], **fitParams)

iterations = 3

for i in range(iterations - 1):
    fqi.partial_fit(None, None, **fitParams)

if env_name == "LQG1D" or env_name == "LQG1DD":
    initial_states = np.zeros((5, 1))
    initial_states[:, 0] = np.linspace(-10,10,5)
    score, stdScore, step, stdStep = evaluate.evaluate_policy(mdp, fqi, 5,
                                                              initial_states=initial_states)
elif env_name == "Acrobot":
    initial_states = np.zeros((5, 4))
    initial_states[:, 0] = np.linspace(-2, 2, 5)
    score, stdScore, step, stdStep = evaluate.evaluate_policy(mdp, fqi, 5,
                                                              initial_states=initial_states)
elif env_name == "SwingPendulum":
    initial_states = np.zeros((5, 2))
    initial_states[:, 0] = np.linspace(-np.pi, np.pi, 5)
    score, stdScore, step, stdStep = evaluate.evaluate_policy(mdp, fqi, 5,
                                                              initial_states=initial_states)
elif env_name == "Bicycle":
    initial_states = np.zeros((1, 5))
    score, stdScore, step, stdStep = evaluate.evaluate_policy(mdp, fqi, 1,
                                                              initial_states=initial_states)
print(score)