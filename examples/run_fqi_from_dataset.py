import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.ensemble import ExtraTreesRegressor
from ifqi import envs
from ifqi.algorithms.fqi import FQI
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_data_for_fqi
from ifqi.models.actionregressor import ActionRegressor
from ifqi.models.regressor import Regressor

"""
Simple script to quickly run fqi from a SARS dataset, where the states have been encoded using the feature extraction module.
This script was specifically created to test the feature extraction module on Atari environments, so it might need fine tuning for other envs.
"""

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true', help='run in debug mode (no output files)')
parser.add_argument('--njobs', type=int, default=-1,
                    help='number of processes to use, set -1 to use all available cores.')
parser.add_argument('--env', type=str, default='BreakoutDeterministic-v3', help='Atari environment to run')
parser.add_argument('--path', type=str, default='../ifqi/algorithms/selection/feature_extraction/output/breakout/dataset/encoded_dataset.csv',
                    help='path to the hdf5 weights file for the autoencoder')
parser.add_argument('--iterations', type=int, default=100, help='number of iterations to run')
args = parser.parse_args()

# Environment
mdp = envs.Atari(name=args.env)
discrete_actions = mdp.action_space.values
state_dim = 0
action_dim = mdp.action_space.n
reward_dim = 1

# Action regressor of ExtraTreesRegressor for FQI
regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'input_scaled': False,
                    'output_scaled': False,
                    'n_jobs': args.njobs}
regressor = Regressor(ExtraTreesRegressor, **regressor_params)
regressor = ActionRegressor(regressor, discrete_actions=discrete_actions, decimals=5, **regressor_params)

# Read dataset
dataset = np.loadtxt(args.path, skiprows=1, delimiter=',')
check_dataset(dataset, state_dim, action_dim, reward_dim) # Can be removed in experiments
print('Dataset has %d samples' % dataset.shape[0])

# Split dataset
sast, r = split_data_for_fqi(dataset, state_dim, action_dim, reward_dim)

# FQI
fqi = FQI(estimator=regressor,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=mdp.gamma,
          horizon=args.iterations,
          verbose=True)

# Run FQI
iteration_values = []
fit_params = {}
fqi.partial_fit(sast, r, **fit_params) # Initial fit (see documentation in FQI.fit for more info)
for i in range(args.iterations - 1):
    print('Iteration %d' % i)
    fqi.partial_fit(None, None, **fit_params)

    # Evaluate policy
    print('Evaluating policy...')
    values = evaluation.evaluate_policy(mdp, fqi)
    print(values)
    iteration_values.append(values[0])

    # Plot results
    if i == 1:
        fig1 = plt.figure(1)
        ax = fig1.add_subplot(1, 1, 1)
        h = ax.plot(range(i + 1), iteration_values, 'ro-')
        plt.ylim(min(iteration_values), max(iteration_values))
        plt.xlim(0, i + 1)
        plt.ion()  # Turns on interactive mode
        plt.show()
    elif i > 1:
        h[0].set_data(range(i + 1), iteration_values)
        ax.figure.canvas.draw()
        plt.ylim(min(iteration_values), max(iteration_values))
        plt.xlim(0, i + 1)
        plt.show()
