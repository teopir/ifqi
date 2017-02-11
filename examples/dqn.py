import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import ifqi.envs as envs
from ifqi.envs.lunarLander import LunarLander
from ifqi.envs.swingUpPendulum import SwingUpPendulum
from ifqi.envs.cartPole import CartPole
import ifqi.evaluation.evaluation as evaluate
from ifqi.dqn.DQN import DQN
from ifqi.models.mlp import MLP
from ifqi.models.actionregressor import ActionRegressor
from ifqi.models.regressor import Regressor

from gym.spaces import prng
import gym
import random

import argparse


parser = argparse.ArgumentParser(
    description='DQN algorithm')

parser.add_argument("experimentName",type=str, help="Provide the name of the experiment")
parser.add_argument("dataset", type=int, help="Provide the index of the dataset")

args = parser.parse_args()

experimentName = args.experimentName
datasetN = args.dataset

act = "relu"

#mdp = CartPole()
#mdp.x_random = False
mdp = LunarLander()
discrete_actions = mdp.action_space.values


mdp.seed(datasetN)
prng.seed(datasetN)
np.random.seed(datasetN)
random.seed(datasetN)

state_dim, action_dim = envs.get_space_info(mdp)

regressor_params = {"n_input": state_dim,#+action_dim,
                    "n_output": 1,
                    "optimizer": "rmsprop",
                     "early_stopping":False,
                     "activation": act,
                     "hidden_neurons":[ 30]*2}


"""regressor_params["input_scaled"]= False
regressor_params["output_scaled"]= False"""
regressor = ActionRegressor(model=MLP, discrete_actions=discrete_actions,decimals=1, **regressor_params)


state_dim, action_dim = envs.get_space_info(mdp)
reward_idx = state_dim + action_dim

mdp.x_random=False
dqn = DQN(env=mdp,
          regr=regressor,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=0.99,
          epsilon=1.,
          epsilon_disc=0.975,
          batch_size=200)

for _ in range(500):
    while not dqn.step():
        pass

dqn.save(experimentName, datasetN)
"""
dqn.env.env.close()
gym.upload("/tmp/gym-results", api_key="sk_iydMqnBQ4uSb4EwG7nJ7Q")
"""
