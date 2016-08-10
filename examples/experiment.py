from __future__ import print_function
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

sys.path.append(os.path.abspath('../'))

from ifqi.experiment import Experiment
from ifqi.fqi.FQI import FQI
import ifqi.utils.parser as parser

# Python 2 and 3: forward-compatible
#from builtins import range


"""
This script can be used to launch experiments using settings
provided in a json configuration file.
The script computes and save the performance of the algorithm
and model in the selected environment averaging on different
experiments and different datasets. While the loop over experiments
is likely to be used for every test, the loop over dataset is not.
Indeed one could prefer to iterate over different number of FQI
steps and so on.

"""
if __name__ == '__main__':
    config_file = './MountainCar/exp3.json'

    exp = Experiment(config_file)

    if 'MLP' in exp.config['model']['model_name']:
        fit_params = {'nb_epoch': exp.config['supervised_algorithm']['n_epochs'],
                      'batch_size': exp.config['supervised_algorithm']['batch_size'],
                      'validation_split': exp.config['supervised_algorithm']['validation_split'],
                      'verbose': exp.config['supervised_algorithm']['verbosity']
                      }
    else:
        fit_params = dict()

    score = np.zeros((exp.config['experiment_setting']['n_experiments'],
                      exp.config['experiment_setting']['n_datasets']))

    for d in range(exp.config['experiment_setting']['n_datasets']):
        print('Dataset: ' + str(d))
        data, state_dim, action_dim, reward_dim = parser.parseReLeDataset(
            '../dataset/' + exp.config['experiment_setting']['load_path'] + str(d) + '.log')
        assert(state_dim == exp.mdp.state_dim)
        assert(action_dim == exp.mdp.action_dim)
        assert(reward_dim == 1)
    
        rewardpos = state_dim + action_dim
        indicies = np.delete(np.arange(data.shape[1]), rewardpos)
    
        sast = data[:, indicies]

        r = data[:, rewardpos]

        for e in range(exp.config['experiment_setting']['n_experiments']):
            print('Experiment: ' + str(e))
    
            fqi = FQI(estimator=exp.model,
                      state_dim=state_dim,
                      action_dim=action_dim,
                      discrete_actions=exp.mdp.n_actions,
                      gamma=exp.config['rl_algorithm']['gamma'],
                      horizon=exp.config['rl_algorithm']['horizon'],
                      verbose=exp.config['rl_algorithm']['verbosity'],
                      scaled=exp.config['rl_algorithm']['scaled'])
                      
            fqi.partial_fit(sast, r, **fit_params)
            for t in range(1, exp.config['rl_algorithm']['n_iterations']):
                fqi.partial_fit(None, None, **fit_params)
                
            score[e, d] = exp.mdp.evaluate(fqi)

    np.save(exp.config['experiment_setting']['save_path'], score)
