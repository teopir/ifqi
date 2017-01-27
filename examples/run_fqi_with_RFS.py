from __future__ import print_function
import matplotlib
matplotlib.use('Agg')  # Necessary to generate plots on headless servers (keep before other imports)
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import time
import sys
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
This script performs Recursive Feature Selection on a SARS dataset collected on an Atari environment,
    and encoded using the feature extraction model passed as parameter (see
    ifqi.algorithms.selection.feature_extraction).
It then attempts to learn the optimal policy by performing Fitted Q-Iteration on the subset of SARS
    tuples selected by RFS.

The SARS dataset can be passed as a file or collected at runtime using the feature extraction
    model, and the RFS selection can be skipped (all features will be used to run FQI).
The feature extraction model can be any object that provides a 'flat_encode' method.
    For now it has been hardcoded to be an Autoencoder object from
    ifqi.algorithms.selection.feature_extraction.Autoencoder, so make sure to remove the mandatory
    flag from the code if you wish to change it.
Note: the Autoencoder must have same architecture and weights that were used to collect the dataset,
    if the dataset was passed as a file.

This script was specifically created to test the feature extraction module on Atari environments,
    so it might need fine tuning for other envs.

Parameters:
    --path (str): path to the hdf5 weights file to initialize the Autoencoder.
    -d, --debug: run the script in debug mode (no output files).
    --log: redirect sys.stdout to a log file in the run folder.
    --njobs (int, -1): number of processes to use (default: use all available).
    --env (str, 'BreakoutDeterministic-v3'): Atari environment to user for collecting the dataset.
    --episodes (int, 100): number of episodes to collect in the dataset.
    --dataset (str, None): path to SARS dataset with encoded state features. Make sure that the
        architecture and weights of the Autoencoder used to collect the dataset are the same
        that are being used now.
    --rfs: perform RFS on the SARS dataset.
    --save-rfs: save the generated RFS dataset.
    --significance (float, 0.1): significance parameter for RFS.
    --iterations (int, 100): number of FQI iterations to run.
"""

### SETUP ###
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='path to the hdf5 weights file for the autoencoder')
parser.add_argument('-d', '--debug', action='store_true', help='run in debug mode (no output files)')
parser.add_argument('--log', action='store_true', help='redirect stdout to file')
parser.add_argument('--njobs', type=int, default=-1, help='number of processes to use (default: use all available)')
parser.add_argument('--env', type=str, default='BreakoutDeterministic-v3', help='Atari environment to run')
parser.add_argument('--episodes', type=int, default=100, help='number of episodes to save in the dataset')
parser.add_argument('--dataset', type=str, default=None, help='path to SARS dataset with encoded features')
parser.add_argument('--rfs', action='store_true', help='perform RFS on the dataset')
parser.add_argument('--save-rfs', action='store_true', help='save the RFS dataset')
parser.add_argument('--significance', type=int, default=0.1, help='significance for RFS')
parser.add_argument('--iterations', type=int, default=100, help='number of FQI iterations to run')
args = parser.parse_args()
logger = Logger(debug=args.debug, output_folder='../ifqi/algorithms/selection/feature_extraction/output/')
# Redirect stdout
if args.log:
    old_stdout = sys.stdout
    sys.stdout = open(logger.path + 'run_log.txt', 'w')

# Create environment
mdp = envs.Atari(name=args.env)

# Feature extraction model (used for collecting episodes and evaluation)
AE = Autoencoder((4, 84, 84), load_path=args.path)

# Read or collect dataset
if args.dataset is not None:
    # Load from disk if dataset was provided
    print('Loading dataset at %s' % args.dataset)
    dataset = np.loadtxt(args.dataset, delimiter=',')
else:
    print('Collecting episodes using model at %s' % args.path)
    collection_params = {'episodes': args.episodes,
                         'env_name': args.env,
                         'header': None,
                         'video': False,
                         'n_jobs': 1}  # n_jobs forced to 1 because AE runs on GPU
    dataset = collect_encoded_dataset(AE, **collection_params)

### ALGORITHMS ###
### RFS ###
reward_dim = 1  # Reward has fixed size of 1
if args.rfs:
    print('Selecting features with RFS...')
    rfs_time = time.time()
    dataset = dataset[1:]  # Remove header
    # Datset parameters for RFS
    action_dim = mdp.action_space.n  # Assuming one hot encoding of the actions
    state_dim = (len(dataset[0][:]) - action_dim - reward_dim - 2) / 2  # Number of state features

    check_dataset(dataset, state_dim, action_dim, reward_dim)
    print('Dataset has %d samples' % dataset.shape[0])

    # Create IFS and RFS models
    ifs_regressor_params = {'n_estimators': 150,
                            'n_jobs': args.njobs}
    ifs_params = {'estimator': ExtraTreesRegressor(**ifs_regressor_params),
                  'n_features_step': 1,
                  'cv': None,
                  'scale': True,
                  'verbose': 1,
                  'significance': args.significance}
    selector = IFS(**ifs_params)

    features_names = np.array(['S%s' % i for i in xrange(state_dim)] + ['A%s' % i for i in xrange(action_dim)])
    rfs_params = {'feature_selector': selector,
                  'features_names': features_names,
                  'verbose': 1}
    fs = RFS(**rfs_params)

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
        selected_actions.extend(range(action_dim))

    # Build dataframe for easy dataset reduction
    header = ['S%s' % i for i in xrange(state_dim)] + ['A%s' % i for i in xrange(action_dim)] + \
             ['R'] + ['SS%s' % i for i in xrange(state_dim)] + ['Absorbing', 'Finished']
    dataframe = pd.DataFrame(dataset, columns=header)
    # Convert actions from onehot to original discrete space
    reverse_onehot_actions = [np.where(dataset[i][state_dim:state_dim + action_dim] == 1)[0][0]
                              for i in xrange(len(dataset))]
    dataframe['A'] = pd.Series(reverse_onehot_actions)

    # Reduce dataset
    support = selected_states + ['A', 'R'] + ['S' + s for s in selected_states] + ['Absorbing', 'Finished']
    reduced_dataset = dataframe[support]  # Remove useless states and one-hot encoding
    reduced_dataset = reduced_dataset[reduced_dataset['A'].isin(selected_actions)]  # Remove useless actions
    reduced_dataset = reduced_dataset.as_matrix()

    # Save RFS tree
    tree = fs.export_graphviz(filename=logger.path + 'rfs_tree.gv')
    tree.save()
    tree.format = 'pdf'
    # TODO I wasn't sure which was the right method to save to pdf
    tree.render()
    tree.save()

    # Save RFS dataset
    if args.save_rfs:
        np.savetxt(logger.path + 'rfs_dataset.csv', reduced_dataset, fmt='%s', delimiter=',')

    rfs_time = time.time() - rfs_time  # Save this for logging
    print('Done RFS. Elapsed time: %s' % rfs_time)
else:
    header = list(dataset[0])
    reduced_dataset = dataset[1:]  # Remove header (the dataset is already reduced)
    action_idx = header.index('A')  # Assuming monodimensional, discrete action
    selected_states = header[:action_idx]  # Selected states are all states
    selected_actions = list(set(reduced_dataset[action_idx, :]))

print('Reduced dataset has %d samples' % reduced_dataset.shape[0])
print('Selected states: %s' % selected_states)
print('Selected actions: %s' % selected_actions)

### FQI ###
# Dataset parameters for FQI
selected_state_dim = len(selected_states)
selected_action_dim = 1  # Assuming monodimensional, discrete action
selected_discrete_actions = np.array(selected_actions)

# Split dataset for FQI
sast, r = split_data_for_fqi(reduced_dataset, selected_state_dim, selected_action_dim, reward_dim)

# Action regressor of ExtraTreesRegressor for FQI
fqi_regressor_params = {'n_estimators': 50,
                        'criterion': 'mse',
                        'min_samples_split': 5,
                        'min_samples_leaf': 2,
                        'input_scaled': False,
                        'output_scaled': False,
                        'n_jobs': args.njobs}
regressor = Regressor(regressor_class=ExtraTreesRegressor,
                      **fqi_regressor_params)
regressor = ActionRegressor(regressor,
                            discrete_actions=selected_discrete_actions,
                            tol=0.5,
                            **fqi_regressor_params)

# Create FQI model
fqi_params = {'estimator': regressor,
              'state_dim': selected_state_dim,
              'action_dim': selected_action_dim,
              'discrete_actions': selected_discrete_actions,
              'gamma': mdp.gamma,
              'horizon': args.iterations,
              'verbose': True}
fqi = FQI(**fqi_params)

# Run FQI
print('Running FQI...')
print('Evaluating policy using model at %s' % args.path)
fqi_time = time.time()

average_episode_duration = len(dataset) / np.sum(dataset[:, -1])
iteration_values = []  # Stores performance of the policy at each step
fqi_fit_params = {}  # Optional parameters for fitting FQI
fqi_evaluation_params = {'metric': 'average',
                         'n_episodes': 1,
                         'selected_states': selected_states,
                         'max_ep_len': 2 * average_episode_duration}

# Fit FQI
fqi.partial_fit(sast, r, **fqi_fit_params)  # Initial fit (see documentation in FQI.fit for more info)
for i in range(args.iterations - 1):
    # print('Iteration %d' % i)
    fqi.partial_fit(None, None, **fqi_fit_params)

    # Evaluate policy
    values = evaluation.evaluate_policy_with_FE(mdp, fqi, AE, **fqi_evaluation_params)
    print(values)
    iteration_values.append(values[0])

fqi_time = time.time() - fqi_time  # Save this for logging
print('Done FQI. Elapsed time: %s' % fqi_time)


### WRITE OUTPUT ###
# Plot iteration values
fig1 = plt.figure(1)
ax = fig1.add_subplot(1, 1, 1)
x = np.array(range(args.iterations - 1))
y = np.array(iteration_values)
h = ax.plot(x, y, 'ro-')
plt.ylim(min(iteration_values), max(iteration_values))
plt.xlim(0, len(x) + 1)
plt.ion()  # Turns on interactive mode
plt.savefig(logger.path + 'fqi_iteration_values.png')

# Log all information of the run for reproducibility
# Restore stdout and close logfile
if args.log:
    sys.stdout.flush()
    sys.stdout.close()
    sys.stdout = old_stdout

print('Logging run information...')
logger.log('### RUN INFORMATION ###')
logger.log(logger.path)

logger.log('\n### FEATURE EXTRACTION ###')
logger.log('AE path: %s' % args.path)

logger.log('\n### DATASET ###')
if args.dataset is not None:
    logger.log('Dataset: %s' % args.dataset)
else:
    logger.log('Dataset collection parameters')
    logger.log(collection_params)

logger.log('\n### RFS ###')
if args.rfs:
    logger.log('Elapsed time: %s' % rfs_time)
    logger.log('IFS regressor parameters')
    logger.log(ifs_regressor_params)
    logger.log('IFS parameters')
    logger.log(ifs_params)
    logger.log('RFS parameters')
    logger.log(rfs_params)
    logger.log('Dataset for RFS')
    logger.log({'action_dim': action_dim,
                'state_dim': state_dim,
                'reward_dim': reward_dim,
                'dataset_size': len(dataset)})
logger.log('reduced_dataset_size: %s' % len(reduced_dataset))
logger.log('Selected states (total %d): \n%s' % (len(selected_states), selected_states))
logger.log('Selected actions (total %d): \n%s' % (len(selected_actions), selected_actions))

logger.log('### FQI ###')
logger.log('Elapsed time: %s' % fqi_time)
logger.log('FQI regressor parameters')
logger.log(fqi_regressor_params)
logger.log('FQI parameters')
logger.log(fqi_params)
logger.log('FQI fit parameters')
logger.log(fqi_fit_params)
logger.log('FQI evaluation parameters')
logger.log(fqi_evaluation_params)

logger.log('(if something isn\'t listed here, it was left as default)')
print('Done.')
