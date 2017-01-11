import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from ifqi import envs
from ifqi.algorithms.fqi import FQI
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import split_data_for_fqi
from ifqi.loadexperiment import get_MDP, get_model
from ifqi.models.actionregressor import ActionRegressor
from ifqi.models.ensemble import Ensemble
from ifqi.models.regressor import Regressor

"""
Script to run fqi. It reads a configuration .json file and perform evaluation
according to the required measure (e.g. dataset size or FQI steps).
"""


def evaluate(mdp, fqi, initial_states, args):
    values = evaluation.evaluate_policy(mdp, fqi,
                                        initial_states=initial_states)
    iteration_values = list()
    iteration_values.append(values[0])

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

    return values

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

results = list()
# Run
for e in range(config['experiment_setting']['evaluation']['n_experiments']):
    print('Experiment: %d' % (e + 1))
    experiment_results = list()

    # Load dataset
    dataset = evaluation.collect_episodes(
        mdp, n_episodes=np.sort(config['experiment_setting']['evaluation']
                                ['n_episodes'])[-1])
    print('Dataset has %d samples' % dataset.shape[0])

    # Load FQI
    fqi = FQI(estimator=regressor,
              state_dim=state_dim,
              action_dim=action_dim,
              discrete_actions=discrete_actions,
              gamma=config['fqi']['gamma'],
              horizon=config['fqi']['horizon'],
              verbose=config['fqi']['verbose'])
    fit_params = config['fit_params']

    if config['experiment_setting']['evaluation']['metric'] == 'n_episodes':
        for i in config['experiment_setting']['evaluation']['n_episodes']:
            episode_end_idxs = np.argwhere(dataset[:, -1] == 1).ravel()
            last_el = episode_end_idxs[i - 1]
            sast, r = split_data_for_fqi(dataset, state_dim, action_dim,
                                         reward_dim, last_el + 1)

            fqi.fit(sast, r, **fit_params)

            experiment_results.append(evaluate(mdp, fqi, mdp.initial_states, args))
        results.append(experiment_results)
    elif config['experiment_setting']['evaluation']['metric'] == 'fqi_iteration':
        sast, r = split_data_for_fqi(dataset, state_dim, action_dim, reward_dim)

        fqi.partial_fit(sast, r, **fit_params)

        for i in range(2, fqi.horizon + 1):
            fqi.partial_fit(None, None, **fit_params)

            if not i % config['experiment_setting']['evaluation']['n_steps_to_evaluate']:
                experiment_results.append(evaluate(mdp, fqi, mdp.initial_states, args))
        results.append(experiment_results)
    else:
        raise ValueError('unknown metric requested.')

if not os.path.exists('results'):
    os.mkdir('results')
np.save('results/' + config['experiment_setting']['save_path'], results)
