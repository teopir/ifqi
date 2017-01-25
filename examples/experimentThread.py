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
parser.add_argument("-s","--loss", action="store_true", help="Provide info on the loss / ect. Valid only for MLP")
parser.add_argument("-r","--screen", action="store_true", help="Screen at the last iteration")
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

haveLoss = args.loss

screen = args.screen

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
experienceEveryNIteration = False
replace = False
mBias = False
if "experienceReplay" in exp.config['experimentSetting']:
    print("expRepl=True")
    experienceReplay = True
    #nExperience = exp.config['rlAlgorithm']["experienceReplay"]['nExperience']
    experienceEveryNIteration = exp.config['rlAlgorithm']["experienceReplay"]['everyNIterations']
    epsilon = exp.config['rlAlgorithm']["experienceReplay"]['epsilon']
    if "replace" in exp.config['rlAlgorithm']["experienceReplay"]:
        replace = exp.config['rlAlgorithm']["experienceReplay"]["replace"]
    if "mBias" in exp.config['rlAlgorithm']["experienceReplay"]:
        mBias = exp.config['rlAlgorithm']["experienceReplay"]["mBias"]

firstReplay = False
if "firstReplay" in exp.config['experimentSetting']:
    firstReplay = exp.config["experimentSetting"]["firstReplay"]

# Here I take the right size
size = exp.config['experimentSetting']['sizes'][sizeN]

environment = exp.getMDP()
environment.seed(datasetN)
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


if "experienceReplay" in exp.config['experimentSetting']:
    if "mBias" in exp.config['rlAlgorithm']["experienceReplay"]:
        nEstDS = exp.config["regressors"][0]["nEstimators"] * 1. / sast.shape[0]

print("Do we have rFirst> 100? ", np.argwhere(rFirst>=100).shape[0])
initial_size = sast.shape[0]
print("n rows:",sast.shape[0])
print(sast[:10])
print(rFirst[:10])

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

    z = varSetting.loadSingle(regressorN, sizeN, datasetN, repetition, iterations, "score")
    #print ("check on",continue_, varSetting.loadSingle(regressorN,sizeN,datasetN,repetition,iterations,"score"))
    if continue_ and (z or type(z)!=bool):
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
                ret_fit = fqi.partial_fit(sastFirst[:], rFirst[:], **fit_params)[2]

                ####################################################################
                # First Experience replay, just for study
                ####################################################################
                if firstReplay:
                    print("first replay")
                    dataset = evaluate.collect_episodes(environment, policy=fqi, n_episodes=500, epsilon=0.1)
                    fqi = exp.getFQI(regressorN)
                    sast_ = np.append(dataset[:, :reward_idx], dataset[:, reward_idx + 1:], axis=1)
                    sastFirst_, rFirst_ = sast_, dataset[:, reward_idx]
                    sast = np.append(sast,sast_,axis=0)
                    rFirst = np.append(rFirst,rFirst_,axis=0)
                    print("new dataset rows: ", sast.shape[0])
                    #raw_input("press any key and then enter")
                    ret_fit = fqi.partial_fit(sast[:], rFirst[:], **fit_params)[2]
                ####################################################################
                # End study
                ####################################################################

            else:
                if iteration > 1 and experienceReplay and iteration% experienceEveryNIteration ==0:
                    print("experience replay")
                    collectionT1 = time.time()
                    dataset = evaluate.collect_episodes(environment, policy=fqi, n_episodes=250, epsilon=epsilon)
                    collectionT2 = time.time()
                    print("episodes collected in: ", collectionT2 - collectionT1 )


                    sast_ = np.append(dataset[:, :reward_idx], dataset[:, reward_idx + 1:], axis=1)
                    sastFirst_, rFirst_ = sast_, dataset[:, reward_idx]

                    n_good = np.argwhere(rFirst_ >= 100).shape[0]

                    varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "nGoal", n_good)

                    print("Do we have new> 100? ", n_good)
                    sast = np.append(sast, sast_, axis=0)
                    rFirst = np.append(rFirst, rFirst_, axis=0)
                    if replace:
                        print("--repace")
                        p = np.random.permutation(sast.shape[0])
                        sast = sast[p][:initial_size,:]
                        rFirst = rFirst[p][:initial_size]

                    n_tot_good = np.argwhere(rFirst_ >= 100).shape[0]
                    print("percentage of goals: ", n_tot_good *100. / sast.shape[0] )

                    varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "pGoal",n_tot_good *100. / sast.shape[0]  )
                    print("new dataset rows: ", sast.shape[0])

                    #raw_input("press any key and then enter")
                    if mBias:
                        exp.config["regressors"][0]["nEstimators"] = int(round(nEstDS * sast.shape[0]))
                        print("number of estimators: ", exp.config["regressors"][0]["nEstimators"])

                    fqi = exp.getFQI(regressorN)
                    ret_fit = fqi.partial_fit(sast[:], rFirst[:], **fit_params)[2]

                else:
                    ret_fit = fqi.partial_fit(None, None, **fit_params)[2]


            if haveLoss:
                print ("ret_fit:", ret_fit)
                varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "nEpoch", len(ret_fit.history['loss']))
                varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "loss", ret_fit.history['loss'][-1])
                varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "valLoss", ret_fit.history['val_loss'][-1])

            if easyReplicability:
                varSetting.savePickle(regressorN,sizeN,datasetN,repetition,iteration,"FQI",fqi)

        end_fit = time.time()
        # ----------------------------------------------------------------------
        # Evaluation
        # ----------------------------------------------------------------------

        if iteration % evaluationEveryNIteration == 0 or iteration==1:
            start_eval = time.time()
            if exp.config["mdp"]["mdpName"] == "LQG1D":
                initial_states = np.zeros((1,1))
                initial_states[0,0] = 10.
                score, stdScore, step, stdStep = evaluate.evaluate_policy(environment, fqi, nEvaluation, initial_states=initial_states)
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
            elif exp.config["mdp"]["mdpName"] == "CarOnHill":
                # evaluation initial states
                initial_states = np.zeros((289, 2))
                cont = 0
                for i in range(-8, 9):
                    for j in range(-8, 9):
                        initial_states[cont, :] = [0.125 * i, 0.375 * j]
                        cont += 1
                score, stdScore, step, stdStep = evaluate.evaluate_policy(environment, fqi, 289*nEvaluation,
                                                                          initial_states=initial_states)
            elif exp.config["mdp"]["mdpName"] == "BicycleNavigate":
                score, stdScore, step, stdStep = evaluate.evaluate_policy(environment, fqi, nEvaluation)
                goal = environment._isAtGoal()
            else:
                if screen:
                    score, stdScore, step, stdStep = evaluate.evaluate_policy(environment, fqi, nEvaluation, render=True)
                else:
                    score, stdScore, step, stdStep = evaluate.evaluate_policy(environment, fqi, nEvaluation)
            end_eval = time.time()

            varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "score", score)
            varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "goalen", score)
            varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "stdScore", stdScore)
            varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "step", step)
            varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "evalTime", end_eval- start_eval)
            varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "fitTime", end_fit - start_fit)

            """if screen:
                raw_input("Type something: ")
                evaluate.evaluate_policy(environment, fqi, nEvaluation, render=True)"""
            # ---------------------------------------------------------------------------
            # Q-Value
            # ---------------------------------------------------------------------------
            """if exp.config["mdp"]["mdpName"] == "LQG1D" and datasetN==0:
                xs = np.linspace(-environment.max_pos, environment.max_pos, 60)
                us = np.linspace(-environment.max_action, environment.max_action, 50)

                l = []
                for x in xs:
                    for u in us:
                        v = fqi.evaluate_Q(x, u)
                        l.append([x, u, v])
                tabular_Q = np.array(l)

                varSetting.save(regressorN, sizeN, datasetN, repetition, iteration, "Q", tabular_Q)"""

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


