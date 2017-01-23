from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ifqi import envs
from ifqi.algorithms.fqi import FQI
from ifqi.algorithms.selection import RFS, IFS
from ifqi.algorithms.selection.feature_extraction.Autoencoder import Autoencoder
from ifqi.algorithms.selection.feature_extraction.build_dataset import collect_encoded_dataset
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_data_for_fqi, split_dataset
from ifqi.models.actionregressor import ActionRegressor
from ifqi.models.regressor import Regressor
from sklearn.ensemble import ExtraTreesRegressor


"""
Simple script to quickly run fqi from a SARS dataset, where the states have been encoded using the feature extraction module.
This script was specifically created to test the feature extraction module on Atari environments, so it might need fine tuning for other envs.
"""

### SETUP ###
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true', help='run in debug mode (no output files)')
parser.add_argument('--njobs', type=int, default=-1,
                    help='number of processes to use, set -1 to use all available cores.')
parser.add_argument('--env', type=str, default='BreakoutDeterministic-v3', help='Atari environment to run')
parser.add_argument('--episodes', type=int, default=100, help='number of episodes to run to collect the dataset')
parser.add_argument('--path', type=str, default='../ifqi/algorithms/selection/feature_extraction/output/breakout/model/model.h5',
                    help='path to the hdf5 weights file for the autoencoder')
parser.add_argument('--dataset', type=str, default=None, help='path to SARS dataset with encoded features')
parser.add_argument('--iterations', type=int, default=100, help='number of iterations to run')
args = parser.parse_args()

# Create environment
mdp = envs.Atari(name=args.env)

# Read or collect dataset
if args.dataset is not None:
    # Load from disk if dataset was provided
    dataset = np.loadtxt(args.dataset, skiprows=1, delimiter=',')
else:
    print('Collecting episodes...')
    AE = Autoencoder((4, 84, 84), load_path=args.path)
    collection_params = {'episodes': args.episodes,
                         'env_name': args.env,
                         'header': None,
                         'video': False,
                         'n_jobs':1}  # n_jobs forced to 1 because AE runs on GPU
    dataset = collect_encoded_dataset(AE, **collection_params)


### RFS ###
# Datset parameters for RFS
action_dim = mdp.action_space.n
#action_dim = 2
reward_dim = 1
state_dim = (len(dataset[0][:]) - action_dim - reward_dim - 2) / 2

check_dataset(dataset, state_dim, action_dim, reward_dim)
print('Dataset has %d samples' % dataset.shape[0])

# Create RFS model
selector = IFS(estimator=ExtraTreesRegressor(n_estimators=50, n_jobs=-1),
               n_features_step=1,
               cv=None,
               scale=True,
               verbose=1,
               significance=0.1)
features_names = np.array(['S%s' % i for i in xrange(state_dim)] + ['A%s' % i for i in xrange(action_dim)])
fs = RFS(feature_selector=selector,
         features_names=features_names,
         verbose=1)

# Split dataset for RFS
state, actions, reward, next_states = split_dataset(dataset, state_dim, action_dim, reward_dim)

# Run RFS
fs.fit(state, actions, next_states, reward)
print(fs.get_support())  # These are the selected features

# Reduce the dataset for FQI
support = list(features_names[np.where(fs.get_support())])
selected_states = []
selected_actions = []
for f in support:
    if f.startswith('S'):
        selected_states.append('S' + f)
    if f.startswith('A'):
        selected_actions.append(f)
support.append('R')
support.extend(selected_states)
support.extend(['Absorbing', 'Finished'])

header = ['S%s' % i for i in xrange(state_dim)] + ['A%s' % i for i in xrange(action_dim)] + \
         ['R'] + ['SS%s' % i for i in xrange(state_dim)] + ['Absorbing', 'Finished']
dataframe = pd.DataFrame(dataset, columns=header)
reduced_dataset = dataframe[support].as_matrix()


### FQI ###
# Dataset parameters for FQI
selected_action_dim = len(selected_actions)
selected_discrete_actions = np.array([[0,1] for a in selected_actions])
selected_state_dim = len(selected_states)

# Split dataset for FQI
sast, r = split_data_for_fqi(reduced_dataset, selected_state_dim, selected_action_dim, reward_dim)

# Action regressor of ExtraTreesRegressor for FQI
regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'input_scaled': False,
                    'output_scaled': False,
                    'n_jobs': args.njobs}
regressor = Regressor(ExtraTreesRegressor, **regressor_params)
regressor = ActionRegressor(regressor, discrete_actions=selected_discrete_actions, decimals=5, **regressor_params)

# Create FQI model
fqi = FQI(estimator=regressor,
          state_dim=selected_state_dim,
          action_dim=selected_action_dim,
          discrete_actions=selected_discrete_actions,
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
