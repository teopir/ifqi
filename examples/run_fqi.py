import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from ifqi.loadexperiment import get_MDP, get_model
from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.algorithms.fqi.FQI import FQI
from ifqi.models.regressor import Regressor
from ifqi.models.actionregressor import ActionRegressor

"""
Script to run fqi. It reads a configuration .json file and perform evaluation
according to the required measure (e.g. dataset size or FQI steps).
"""


def evaluate(mdp, fqi, initial_states, args):
    values = evaluation.evaluate_policy(mdp, fqi,
                                        initial_states=initial_states)
    iteration_values = list()
    results = list()
    print('J: %f' % values[0])
    iteration_values.append(values[0])
    results.append(values)

    if args.plot:
        if i == 1:
            fig1 = plt.figure(1)
            ax = fig1.add_subplot(1, 1, 1)
            h = ax.plot(range(i + 1), iteration_values, 'ro-')
            plt.ylim(min(iteration_values), max(iteration_values))
            plt.xlim(0, i + 1)
            plt.ion()  # turns on interactive mode
            plt.show()
        elif i > 1:
            h[0].set_data(range(i + 1), iteration_values)
            ax.figure.canvas.draw()
            plt.ylim(min(iteration_values), max(iteration_values))
            plt.xlim(0, i + 1)
            plt.show()

    return results

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default=None,
                    help='path of configuration file to load')
parser.add_argument('-p', '--plot', action='store_true',
                    default=False, help='plot results during the run.')
args = parser.parse_args()

# Load config file
if args.config is not None:
    load_path = args.config
    with open(load_path) as f:
        config = json.load(f)
else:
    raise ValueError('configuration file path missing.')

# Load environment
mdp = get_MDP(config['mdp']['name'])
mdp.set_seed(config['experiment_setting']['evaluation']['seed'])
state_dim, action_dim, reward_dim = envs.get_space_info(mdp)
assert reward_dim == 1
reward_idx = state_dim + action_dim
discrete_actions = mdp.action_space.values

# Load model
regressor_params = config['model']['params']
regressor_class = get_model(config['model']['name'])
if config['model']['ensemble']:
    regressor = Ensemble(regressor_class, **regressor_params)
else:
    regressor = Regressor(regressor_class, **regressor_params)
if not config['model']['fit_actions']:
    regressor = ActionRegressor(regressor, discrete_actions=discrete_actions,
                                decimals=5, **regressor_params)

# Load FQI
fqi = FQI(estimator=regressor,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=config['fqi']['gamma'],
          horizon=config['fqi']['horizon'],
          features=config['fqi']['features'],
          verbose=config['fqi']['verbose'])
fit_params = config['fit_params']

# Load dataset
dataset = evaluation.collect_episodes(
    mdp, n_episodes=config['experiment_setting']['evaluation']
                          ['n_episodes'][-1])
print('Dataset has %d samples' % dataset.shape[0])

# Load initial state to start evaluation episodes. This is the only setting
# to be chosen outside the configuration file.
# IF MULTIPLE EXPERIMENTS ARE TO BE PERFORMED STARTING FROM THE SAME
# INITIAL STATE, USE AN ARRAY WITH THE SAME INITIAL STATE REPEATED FOR THE
# DESIRED NUMBER OF EVALUATION RUNS.
initial_states = np.zeros((289, 2))
cont = 0
for i in range(-8, 9):
    for j in range(-8, 9):
        initial_states[cont, :] = 0.125 * i + 0.375 * j
        cont += 1
######################################################################
######################################################################

experiment_results = list()
results = list()
# Run
if config['experiment_setting']['evaluation']['metric'] == 'n_episodes':
    for e in range(config['experiment_setting']['evaluation']['n_experiments']):
        for i in config['experiment_setting']['evaluation']['n_episodes']:
            episode_end_idxs = np.argwhere(dataset[:, -1] == 1).ravel()
            last_el = episode_end_idxs[i - 1]
            sast = np.append(dataset[:last_el + 1, :reward_idx],
                             dataset[:last_el + 1, reward_idx + 1:-1],
                             axis=1)
            r = dataset[:last_el + 1, reward_idx]

            fqi.fit(sast, r, **fit_params)

            experiment_results.append(evaluate(mdp, fqi, initial_states, args))
        results.append(experiment_results)
elif config['experiment_setting']['evaluation']['metric'] == 'fqi_iteration':
    sast = np.append(dataset[:, :reward_idx],
                     dataset[:, reward_idx + 1:-1],
                     axis=1)
    r = dataset[:, reward_idx]

    for e in range(config['experiment_setting']['evaluation']['n_experiments']):
        fqi.partial_fit(sast, r, **fit_params)

        for i in range(2, fqi.horizon + 1):
            fqi.partial_fit(None, None, **fit_params)

            if not i % config['experiment_setting']['evaluation']['n_steps_to_evaluate']:
                experiment_results.append(
                    evaluate(mdp, fqi, initial_states, args))
        results.append(experiment_results)
else:
    raise ValueError('unknown metric requested.')

if not os.path.exists('results'):
    os.mkdir('results')
np.save('results/' + config['experiment_setting']['save_path'], results)
