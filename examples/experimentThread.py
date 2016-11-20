# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 22:08:52 2016

@author: samuele
"""

from __future__ import print_function

import os
import sys
import cPickle
import ifqi.envs as envs
from variableLoadSave import ExperimentVariables
import ifqi.evaluation.evaluation as evaluate
from ifqi.experiment import Experiment
import argparse
import time
from gym.spaces import prng

import numpy as np
import time
import random
"""

Single thread of experimentThreadManager
"""

# ------------------------------------------------------------------------------
# Retrive params
# ------------------------------------------------------------------------------


parser = argparse.ArgumentParser(
    description='Execution of one experiment thread provided a configuration file and\n\t A regressor (index)\n\t Size of dataset (index)\n\t Dataset (index)')

parser.add_argument("experimentName",type=str, help="Provide the name of the experiment")
parser.add_argument("configFile", type=str, help="Provide the filename of the configuration file")
parser.add_argument("regressor", type=int, help="Provide the index of the regressor listed in the configuration file")
parser.add_argument("size", type=int, help="Provide the index of the size listed in the configuration file")
parser.add_argument("dataset", type=int, help="Provide the index of the dataset")
parser.add_argument("-c","--cont", action="store_true", help="Provide the index of the dataset")
args = parser.parse_args()

experimentName = args.experimentName

config_file = args.configFile
# Every experiment just run a regressor that is selected by the ExperimentThreadManager, and here we get the index
regressorN = args.regressor
# Every experiment just run a specific dataset size. ExperimentThreadManager select one index of size
sizeN = args.size
# Every experiment just run a specific dataset. ExperimentThreadManager select one specific dataset
datasetN = args.dataset

continue_ = args.cont

print("Started experiment with regressor " + str(regressorN)+ " dataset " + str(datasetN) + ", size " + str(sizeN))


prng.seed(datasetN)
np.random.seed(datasetN)
random.seed(datasetN)

exp = Experiment(config_file)

# ------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------

iterations = exp.config['rlAlgorithm']['nIterations']
repetitions = exp.config['experimentSetting']["nRepetitions"]

nEvaluation = exp.config['experimentSetting']["evaluations"]['nEvaluations']
evaluationEveryNIteration = exp.config['experimentSetting']["evaluations"]['everyNIterations']

saveFQI = False
saveEveryNIteration = 0
if "saveEveryNIteration" in exp.config['experimentSetting']:
    saveFQI = True
    saveEveryNIteration = exp.config['experimentSetting']["saveEveryNIteration"]

easyReplicability = False
if "easyReplicability" in exp.config['experimentSetting']:
    easyReplicability = exp.config['experimentSetting']['easyReplicability']


experienceReplay = False
if "experienceReplay" in exp.config['experimentSetting']:
    experienceReplay = True
    nExperience = exp.config['rlAlgorithm']["experienceReplay"]['nExperience']
    experienceEveryNIteration = exp.config['rlAlgorithm']["experienceReplay"]['everyNIterations']

# Here I take the right size
size = exp.config['experimentSetting']['sizes'][sizeN]

environment = exp.getMDP()
environment.set_seed(datasetN)
#Could be problematic
regressorName = exp.getModelName(regressorN)

#TODO:clearly not a good solution
if(regressorName=="MLP" or regressorName=="MLPEnsemble"):
    fit_params = exp.getFitParams(regressorN)
else:
    fit_params = {}


ds_filename =  ".regressor_" + str(regressorName) + "size_" + str(size) + "dataset_" + str(
    datasetN) + ".npy"
pkl_filename = ".regressor_" + str(regressorName) + "size_" + str(size) + "dataset_" + str(
    datasetN) + ".pkl"
# TODO:
action_dim = 1

# ------------------------------------------------------------------------------
# Dataset Generation
# ------------------------------------------------------------------------------

state_dim, action_dim = envs.get_space_info(environment)
reward_idx = state_dim + action_dim
dataset = evaluate.collect_episodes(environment,policy=None,n_episodes=size)
sast = np.append(dataset[:, :reward_idx], dataset[:, reward_idx + 1:], axis=1)
sastFirst, rFirst = sast, dataset[:, reward_idx]

# ------------------------------------------------------------------------------
# FQI Loading
# ------------------------------------------------------------------------------

actualRepetition = 0
actualIteration = 1

if os.path.isfile(pkl_filename):
    fqi_obj = cPickle.load(open(pkl_filename, "rb"))
    dataset.reset()
    dataset.load()
    fqi = fqi_obj["fqi"]
    actualIteration = fqi_obj["actualIteration"] + 1
    actualRepetition = fqi_obj["actualRepetition"]


# ------------------------------------------------------------------------------
# FQI Iterations
# ------------------------------------------------------------------------------

#We evaluate with a different horizon
if "horizon" in exp.config["mdp"]:
    environment.horizon = exp.config["mdp"]["horizon"]

varSetting = ExperimentVariables(experimentName)
replay_experience = False
for repetition in range(actualRepetition, repetitions):

    #print ("check on",continue_, varSetting.loadSingle(regressorN,sizeN,datasetN,repetition,iterations,"score"))
    if continue_ and varSetting.loadSingle(regressorN,sizeN,datasetN,repetition,iterations,"score"):
        print ("Experiment already done")
        break

    for iteration in range(actualIteration, iterations + 1):

        # ----------------------------------------------------------------------
        # Fit
        # ----------------------------------------------------------------------
        start_fit = time.time()

        fqi_ = False
        if easyReplicability:
            fqi_ = varSetting.loadPickle(regressorN, sizeN, datasetN, repetition, iteration, "FQI")

        if fqi_:
            fqi = fqi_
        else:

            if iteration==1:
                fqi = exp.getFQI(regressorN)
                fqi.partial_fit(sastFirst[:], rFirst[:], **fit_params)
            else:
                if replay_experience:
                    #TODO:understand what's wrong
                    fqi.partial_fit(sast[:], r[:], **fit_params)
                    replay_experience = False
                else:
                    fqi.partial_fit(None, None, **fit_params)

            if easyReplicability:
                varSetting.savePickle(regressorN,sizeN,datasetN,repetition,iteration,"FQI",fqi)

        end_fit = time.time()
        # ----------------------------------------------------------------------
        # Evaluation
        # ----------------------------------------------------------------------

        if iteration % evaluationEveryNIteration == 0 or iteration==1:
            start_eval = time.time()
            if exp.config["mdp"]["mdpName"] == "LQG1D":
                score, stdScore, step, stdStep = evaluate.evaluate_policy(environment, fqi, nEvaluation, initialState=10)
            elif exp.config["mdp"]["mdpName"] == "Acrobot":
                initial_states = np.zeros((41, 4))
                initial_states[:, 0] = np.linspace(-2, 2, 41)
                score, stdScore, step, stdStep = evaluate.evaluate_policy(environment, fqi, 41*nEvaluation,
                                                                          initial_states=initial_states)
            elif exp.config["mdp"]["mdpName"] == "SwingPendulum":
                initial_states = np.zeros((21, 2))
                initial_states[:, 0] = np.linspace(-np.pi, np.pi, 21)
                score, stdScore, step, stdStep = evaluate.evaluate_policy(environment, fqi, 21*nEvaluation,
                                                                          initial_states=initial_states)
            else:
                score, stdScore, step, stdStep = evaluate.evaluate_policy(environment, fqi, nEvaluation)
            end_eval = time.time()

            varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "score", score)
            varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "stdScore", stdScore)
            varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "step", step)
            varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "evalTime", end_eval- start_eval)
            varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "fitTime", end_fit - start_fit)

            # ---------------------------------------------------------------------------
            # Q-Value
            # ---------------------------------------------------------------------------
            if exp.config["mdp"]["mdpName"] == "LQG1D" and datasetN==0:
                xs = np.linspace(-environment.max_pos, environment.max_pos, 60)
                us = np.linspace(-environment.max_action, environment.max_action, 50)

                l = []
                for x in xs:
                    for u in us:
                        v = fqi.evaluate_Q(x, u)
                        l.append([x, u, v])
                tabular_Q = np.array(l)

                varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "Q", tabular_Q)

        # ----------------------------------------------------------------------
        # SAVE FQI STATUS
        # ----------------------------------------------------------------------

        if saveFQI:
            if iteration % saveEveryNIteration == 0:
                directory = os.path.dirname(pkl_filename)
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                cPickle.dump({'fqi': fqi, 'actualIteration': iteration, "actualRepetition": repetition},
                             open(pkl_filename, "wb"))
                dataset.save(ds_filename)



    actualIteration = 0

