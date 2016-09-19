from __future__ import print_function
import os
import sys
import time
import json
from time import gmtime, strftime

from context import *

from ifqi.experiment import Experiment
import subprocess
from random import shuffle

# Python 2 and 3: forward-compatible
#from builtins import range


"""
provided in a json configuration file.
The script computes and save the performance of the algorithm
and model in the selected environment averaging on different
experiments and different datasets. While the loop over experiments
is likely to be used for every test, the loop over dataset is not.
Indeed one could prefer to iterate over different number of FQI
steps and so on.

This version allow multithreading

"""
def execute(commands, nThread, refresh_time=10.,shuffled=False):

        if(shuffled):
            shuffle(commands)
            
        processes = set()
        #command should be a list
        try:
            for command in commands:
                print ("New Thread Executed")
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
    config_file = "results/" + sys.argv[1]
    nThread = int(sys.argv[2])
    add_last = sys.argv[3]
    exp = Experiment(config_file)

    if not add_last=='True':
        name = raw_input("Please define the name of the new experiment")
        description = raw_input("Please write a description of the experiment")
        
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
    
    execute(commands,nThread,10.)
    
    #--------------------------------------------------------------------------
    # DiaryExperiment
    #--------------------------------------------------------------------------
    
    
    
    try:
        with open("results/diary.json", 'r') as fp:
            diary = json.load(fp)
    except:
        diary = []
    
    if add_last=='True':
        last_exp = diary[-1]
        last_exp["jsonFile"].append(config_file)
        diary[-1] = last_exp
    else:
        diaryExperiment = {
        "name":name,
        "date":strftime("%d-%m-%Y %H:%M:%S", gmtime()),
        "description":description,
        "jsonFile":[config_file],
        "images":[],
        "postComment":"",
        "importance":""
        }
        diary.append(diaryExperiment)
    
    with open("results/diary.json", 'w') as fp:
        json.dump(diary,fp)
    
    
    
