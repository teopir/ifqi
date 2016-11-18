from __future__ import print_function
import os
import sys
import time
import json
from time import gmtime, strftime

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

import argparse

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

    parser = argparse.ArgumentParser(
        description="""Welcome in experimentLauncher
        Here we will launch the experiment and we will (if required) include it in the diary.json
        """)

    parser.add_argument("experimentName", type=str, help="Provide the name of the experiment")
    parser.add_argument("configFile", type=str, help="Provide the name of the json file")
    parser.add_argument("-d", "--diary", help="The experiment will be included in the diary", action="store_true")
    parser.add_argument("-l", "--addLast", help="The experiment will be merged with the last one present in the diary", action="store_true")
    parser.add_argument("-c", "--cont", help="continue the experiment made", action="store_true")
    parser.add_argument("nThread", type=int, help="Set the number of cores")

    args = parser.parse_args()

    experimentName = args.experimentName
    configFile = args.configFile
    diaryFlag = args.diary
    nThread = args.nThread
    addLast = args.addLast
    continue_ = args.cont
    exp = Experiment(configFile)

    commands = []

    myPath = os.path.realpath(__file__)
    myPath = os.path.dirname(myPath)
    myPath += "/experimentThread.py"
    for regressor in range(len(exp.config["regressors"])):
        for size in range(len(exp.config["experimentSetting"]["sizes"])):
            for dataset in range(exp.config['experimentSetting']['datasets']):
                if continue_:
                    commands.append(["python", myPath ,experimentName, configFile, str(regressor),  str(size), str(dataset), "--cont"])
                else:
                    commands.append(["python", myPath, experimentName, configFile, str(regressor), str(size), str(dataset)])
    
    execute(commands,nThread,10.)
    
    #--------------------------------------------------------------------------
    # DiaryExperiment
    #--------------------------------------------------------------------------

    if diaryFlag:
        try:
            with open("diary.json", 'r') as fp:
                diary = json.load(fp)
        except:
            diary = []

        if addLast=='True':
            last_exp = diary[-1]
            last_exp["jsonFile"].append(configFile)
            diary[-1] = last_exp
        else:
            description = raw_input("Prompt a description of the experiment please: ")
            stars = raw_input("Prompt a  number from 0 to 5 about the importance of this experiment: ")
            diaryExperiment = {
            "name":experimentName,
            "date":strftime("%d-%m-%Y %H:%M:%S", gmtime()),
            "description":description,
            "jsonFile":[configFile],
            "images":[],
            "postComment":"",
            "importance":str(stars)
            }
            diary.append(diaryExperiment)

        with open("diary.json", 'w') as fp:
            json.dump(diary,fp)
    
    
    
