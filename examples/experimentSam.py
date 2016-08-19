from __future__ import print_function
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

sys.path.append(os.path.abspath('../'))

from ifqi.experiment import Experiment
from ifqi.fqi.FQI import FQI
import ifqi.utils.parser as parser
import threading 

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

def experiment_unit(exp,d,e):
    data, state_dim, action_dim, reward_dim = parser.parseReLeDataset(
            '../dataset/' + exp.config['experiment_setting']['load_path'] + "/"+ d + '.txt')
    assert(state_dim == exp.mdp.state_dim)
    assert(action_dim == exp.mdp.action_dim)
    assert(reward_dim == 1)

    rewardpos = state_dim + action_dim
    indicies = np.delete(np.arange(data.shape[1]), rewardpos)

    sast = data[:, indicies]

    r = data[:, rewardpos]
    print('Experiment: ' + str(e))
            
    exp.loadModel()
    
    if 'features' in exp.config['model']:
        features = exp.config['model']['features']
    else:
        features = None
        
    fqi = FQI(estimator=exp.model,
              state_dim=state_dim,
              action_dim=action_dim,
              discrete_actions=exp.mdp.n_actions,
              gamma=exp.config['rl_algorithm']['gamma'],
              horizon=exp.config['rl_algorithm']['horizon'],
              verbose=exp.config['rl_algorithm']['verbosity'],
              features=features,
              scaled=exp.config['rl_algorithm']['scaled'])
              
    fqi.partial_fit(sast, r, **fit_params)
    for t in range(1, exp.config['rl_algorithm']['n_iterations']):
        fqi.partial_fit(None, None, **fit_params)
        if "save_iteration"  in exp.config['experiment_setting']:
            if t % exp.config['experiment_setting']["save_iteration"] == 0:
                score, step, goal = exp.mdp.evaluate(fqi)
                dir_ =exp.config["experiment_setting"]["save_path"]
                directory =os.path.dirname(dir_+ "/" + "score_" + d + "_" + str(e) + "_" + str(t) + ".npy")
                if not os.path.isdir(directory): os.makedirs(directory)
                directory =os.path.dirname(dir_+ "/" + "step_" + d + "_" + str(e) + "_" + str(t) + ".npy")
                if not os.path.isdir(directory): os.makedirs(directory)
                directory =os.path.dirname(dir_+ "/" + "goal_" + d + "_" + str(e) + "_" + str(t) + ".npy")
                if not os.path.isdir(directory): os.makedirs(directory)
                np.save(dir_+ "/" + "score_" + d + "_" + str(e) + "_" + str(t), score)
                np.save(dir_+ "/" + "step_" + d + "_" + str(e) + "_" + str(t), step)
                np.save(dir_+ "/" + "goal_" + d + "_" + str(e) + "_" + str(t), goal)
        
    score, step, goal = exp.mdp.evaluate(fqi)
    dir_ =exp.config["experiment_setting"]["save_path"]
    if not os.path.isdir(dir_): os.makedirs(dir_)
    np.save(dir_+ "/" + "score_" + d + "_" + str(e) + "_last", score)
    np.save(dir_+ "/" + "step_" + d + "_" + str(e) + "_last", step)
    np.save(dir_+ "/" + "goal_" + d + "_" + str(e) + "_last", goal)

if __name__ == '__main__':
    config_file = sys.argv[1]

    exp = Experiment(config_file)

    if 'MLP' in exp.config['model']['model_name']:
        fit_params = {'nb_epoch': exp.config['supervised_algorithm']['n_epochs'],
                      'batch_size': exp.config['supervised_algorithm']['batch_size'],
                      'validation_split': exp.config['supervised_algorithm']['validation_split'],
                      'verbose': exp.config['supervised_algorithm']['verbosity']
                      }
    else:
        fit_params = dict()

    """score = np.zeros((exp.config['experiment_setting']['n_experiments'],
                      exp.config['experiment_setting']['n_datasets']))"""
    jobs = []
    started_job=set()
    for d in exp.config['experiment_setting']['datasets']:
        for e in range(exp.config['experiment_setting']['n_experiments']):
            experiment_unit(exp,d,e)