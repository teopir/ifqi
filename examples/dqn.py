import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import ifqi.envs as envs
from ifqi.envs.lunarLander import LunarLander
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
    description='Execution of one experiment thread provided a configuration file and\n\t A regressor (index)\n\t Size of dataset (index)\n\t Dataset (index)')

act = "relu"

mdp = LunarLander()
discrete_actions = mdp.action_space.values


mdp.seed(0)

state_dim, action_dim = envs.get_space_info(mdp)

regressor_params = {"n_input": state_dim+action_dim,
                    "n_output": 1,
                    "optimizer": "rmsprop",
                     "early_stopping":False,
                     "activation": act,
                     "hidden_neurons":[ 150]*2}


regressor_params["input_scaled"]= False
regressor_params["output_scaled"]= False
regressor = Regressor(regressor_class=MLP, **regressor_params)


state_dim, action_dim = envs.get_space_info(mdp)
reward_idx = state_dim + action_dim

dqn = DQN(env=envs.LunarLander(),
          regr=regressor,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=0.99,
          epsilon=1.,
          batch_size=200)

while True:
    dqn.step()

