from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from ifqi import envs
from ifqi.algorithms.fqi import FQI
from ifqi.algorithms.selection import RFS, IFS
from ifqi.algorithms.selection.feature_extraction.Autoencoder import Autoencoder
from ifqi.algorithms.selection.feature_extraction.build_dataset import collect_encoded_dataset
from ifqi.algorithms.selection.feature_extraction.Logger import Logger
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
parser.add_argument('--path', type=str, default='../ifqi/algorithms/selection/feature_extraction/output/breakout/model/model.h5',
                    help='path to the hdf5 weights file for the autoencoder')
parser.add_argument('--episodes', type=int, default=100, help='number of episodes to run to collect the dataset')
parser.add_argument('--dataset', type=str, default=None, help='path to SARS dataset with encoded features')
parser.add_argument('--rfs', action='store_true', help='perform RFS on the dataset')
parser.add_argument('--save-rfs', action='store_true', help='save the RFS dataset')
parser.add_argument('--iterations', type=int, default=100, help='number of FQI iterations to run')
args = parser.parse_args()
logger = Logger(debug=args.debug, output_folder='../ifqi/algorithms/selection/feature_extraction/output/')

# Create environment
mdp = envs.Atari(name=args.env)

# Read or collect dataset
if args.dataset is not None:
    # Load from disk if dataset was provided
    print('Loading dataset at %s' % args.dataset)
    dataset = np.loadtxt(args.dataset, delimiter=',')
else:
    print('Collecting episodes using model at %s' % args.path)
    AE = Autoencoder((4, 84, 84), load_path=args.path)
    collection_params = {'episodes': args.episodes,
                         'env_name': args.env,
                         'header': None,
                         'video': False,
                         'n_jobs':1}  # n_jobs forced to 1 because AE runs on GPU
    dataset = collect_encoded_dataset(AE, **collection_params)


### RFS ###
reward_dim = 1 # Reward has fixed size of 1
if args.rfs:
    print('Selecting features with RFS...')
    t = time.time()
    dataset = dataset[1:] # Remove header
    # Datset parameters for RFS
    action_dim = mdp.action_space.n # Assuming one hot encoding of the actions
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

    # Reduce the dataset for FQI
    selected_states = []
    selected_actions = []
    for f in features_names[np.where(fs.get_support())]:
        if f.startswith('S'):
            selected_states.append(f)
        if f.startswith('A'):
            selected_actions.append(int(f.lstrip('A')))

    # Remove this during experiments
    if len(selected_actions) == 0:
        selected_actions.extend([1,2])

    # Build dataframe for easy dataset reduction
    header = ['S%s' % i for i in xrange(state_dim)] + ['A%s' % i for i in xrange(action_dim)] + \
             ['R'] + ['SS%s' % i for i in xrange(state_dim)] + ['Absorbing', 'Finished']
    dataframe = pd.DataFrame(dataset, columns=header)
    # Convert actions from onehot to original discrete space
    reverse_onehot_actions = [np.where(dataset[i][state_dim:state_dim+action_dim] == 1)[0][0] for i in xrange(len(dataset))]
    dataframe['A'] = pd.Series(reverse_onehot_actions)

    # Reduce dataset
    support = selected_states + ['A', 'R'] + ['S' + s for s in selected_states] + ['Absorbing', 'Finished']
    reduced_dataset = dataframe[support]  # Remove useless states and one-hot encoding
    reduced_dataset = reduced_dataset[reduced_dataset['A'].isin(selected_actions)]  # Remove useless actions
    reduced_dataset = reduced_dataset.as_matrix()

    # Save RFS tree
    fs.export_graphviz(filename=logger.path + 'rfs_tree.gv')

    # Save RFS dataset
    if args.save_rfs:
        np.savetxt(logger.path + 'rfs_dataset.csv', reduced_dataset, fmt='%s', delimiter=',')

    print('Done RFS. Elapsed time: %s' % (time.time() - t))
else:
    header = list(dataset[0])
    reduced_dataset = dataset[1:] # Remove header (the dataset is already reduced)
    action_idx = header.index('A')
    selected_states = header[:action_idx]
    selected_actions = list(set(reduced_dataset[action_idx, :]))

print('Reduced dataset has %d samples' % reduced_dataset.shape[0])
print('Selected states: %s' % selected_states)
print('Selected actions: %s' % selected_actions)

### FQI ###
# Dataset parameters for FQI
selected_state_dim = len(selected_states)
selected_action_dim = 1 # Assuming monodimensional, discrete action
selected_discrete_actions = np.array(selected_actions)

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
regressor = Regressor(regressor_class=ExtraTreesRegressor, **regressor_params)
regressor = ActionRegressor(regressor, discrete_actions=selected_discrete_actions, tol=0.5, **regressor_params)

# Create FQI model
fqi = FQI(estimator=regressor,
          state_dim=selected_state_dim,
          action_dim=selected_action_dim,
          discrete_actions=selected_discrete_actions,
          gamma=mdp.gamma,
          horizon=args.iterations,
          verbose=True)

# Run FQI
print('Running FQI...')
t = time.time()

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
        plt.savefig(logger.path + 'fqi_%s.png' % i)
    elif i > 1:
        h[0].set_data(range(i + 1), iteration_values)
        ax.figure.canvas.draw()
        plt.ylim(min(iteration_values), max(iteration_values))
        plt.xlim(0, i + 1)
        # plt.show()
        plt.savefig(logger.path + 'fqi_%s.png' % i)

print('Done FQI. Elapsed time: %s' % (time.time() - t))