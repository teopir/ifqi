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
import subprocess
from random import shuffle

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

This version allow multithreading

"""
def execute(commands, nThread, refresh_time=.2,shuffled=True):

        """if(shuffled):
            shuffle(commands)"""
            
        processes = set()
        #command should be a list
        try:
            for command in commands:
                processes.add(subprocess.Popen(command))
                while len(processes) >= nThread:
                    time.sleep(refresh_time)
                    processes.difference_update([p for p in processes if p.poll() is not None])
        
            for p in processes:
                if p.poll() is None:
                    p.wait()
        except KeyboardInterrupt:
            print("Keyboard interrupt catch")
            for p in processes:
                p.kill()
                exit()
        
        print("All sample executed")

if __name__ == '__main__':
    config_file = sys.argv[1]
    nThread = int(sys.argv[2])
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
    commands = []
    for d in exp.config['experiment_setting']['datasets']:
        for e in range(exp.config['experiment_setting']['n_experiments']):
            commands.append(["python","experimentThread.py" , config_file , str(d) ,  str(e)])
    
    execute(commands,nThread,1.)