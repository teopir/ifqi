# -*- coding: utf-8 -*-
"""
This script take different variables saved in numpy, computes the mean and the average and plots them.
This specific plot will show how is J (and steps) throungh FQI iterations
"""

import numpy as np
import json
import os
import os.path
from ifqi.experiment import Experiment
import matplotlib.pyplot as plt
from examples.variableLoadSave import  ExperimentVariables
import json
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import re

"""example of data
data={mlp:{
iteration:[1,2,3,4,5]
mean:[1,2,3,4,5],
std:None
},
wide:{
mean:[2,3,1,4],
std:[0.5,0.5,0.6,0.6]
}}
"""


def multiOneDimPlot(data, xlabel, ylabel, title, path):
    color = ["r", "g", "b", "k"]
    imgName = os.path.realpath(path)
    i = 0
    hand = []
    min_ = np.inf
    max_ = -np.inf
    for dic in data:
        dict_ = data[dic]
        name = dict_["name"]
        mean = dict_["mean"]
        conf = dict_["conf"]

        if min(mean) < min_:
            min_ = min(mean)
        if max(mean) > max_:
            max_ = max(mean)

        if (len(conf) != 0):
            plt.errorbar(dict_["iteration"], mean, color=color[i], yerr=conf, lw=1, label=name)
        hand.append(plt.plot(dict_["iteration"], mean, color=color[i], lw=1, label=name)[0])

        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)

        i += 1
        i %= len(color)

    plt.legend(handles=hand, loc=4)
    boundaries = max(abs(min_) * 0.1, 0.1 * abs(max_))
    plt.ylim(min_ - boundaries, max_ + boundaries)
    plt.savefig(imgName)
    plt.show()

def ask(message, err_message, function):
    # type: (string, string, lambda) -> object
    while True:
        try:
            ret = raw_input(message)
            ret = function(ret)
            print("OK!")
            break
        except KeyboardInterrupt:
            print
            'Interrupted'
            exit()
        except:
            print(err_message)
    return ret


experimentPath = ask("Prompt the folder of your experiment: ", "Please choose a right path", str)
jsonFile = ask("JsonFile of your experiment: ", "Please choose a right path", str)
exp = Experiment(configFile=jsonFile)
expVar = ExperimentVariables(experimentPath)

variables = ["score", "stdScore", "evalTime", "fitTime"]
for variable in variables:
    if input("Would you like to plot variable " + variable + " [True / False]?"):
        get = expVar.getOverallSituation(variable)
        nLines = len(get)
        print(str(nLines) + " lines are found.")

        for s in expVar.sizeLoaded:
            data = {}
            for reg in expVar.regressorLoaded:
                key = (reg, s)
                data[key] = {}
                size = exp.config['experimentSetting']['sizes'][s]
                data[key]["name"] = str(exp.getModelName(reg)) + "_" + str(size)
                data[key]["iteration"] = get[key][0]
                iteration = get[key][0]
                if variable=="stdScore":
                    data[key]["mean"] = np.array(get[key][1]) * np.sqrt(exp.config['experimentSetting']['evaluations']['nEvaluations']) / 1.96
                    data[key]["conf"] = np.array(get[key][2]) * np.sqrt(exp.config['experimentSetting']['evaluations']['nEvaluations']) / 1.96
                else:
                    data[key]["mean"] = get[key][1]
                    data[key]["conf"] = get[key][2]
            filepath = "plot/" + experimentPath + "/" + variable + "_size" + str(size)+".jpg"
            directory =os.path.dirname(filepath)
            if not os.path.isdir(directory): os.makedirs(directory)
            """if variable == "score":
                key = ("opt",s)
                data[key] = {}
                data[key]["name"] = "Optimal score"
                data[key]["iteration"] = iteration
                data[key]["mean"] = [-146.285467395] * len(iteration)
                data[key]["conf"] = [0.75] * len(iteration)"""
            multiOneDimPlot(data,"iterations",variable,"size=" + str(size),filepath)

        get = expVar.getSizeLines(variable)
        data = {}
        for reg in expVar.regressorLoaded:
            key=reg
            data[key] = {}
            data[key]["name"] = str(exp.getModelName(reg))
            data[key]["iteration"] = [exp.config['experimentSetting']['sizes'][x] for x in get[key][0]]
            iteration = data[key]["iteration"]
            data[key]["mean"] = get[key][1]
            data[key]["conf"] = get[key][2]

        """if variable == "score":
            key = ("opt", s)
            data[key] = {}
            data[key]["name"] = "Optimal score"
            data[key]["iteration"] = iteration
            data[key]["mean"] = [-146.285467395] * len(iteration)
            data[key]["conf"] = [0.75] * len(iteration)"""
        filepath = "plot/" + experimentPath + "/" + variable + "_sizeview.jpg"
        directory =os.path.dirname(filepath)
        if not os.path.isdir(directory): os.makedirs(directory)
        multiOneDimPlot(data,"sizes",variable,"SizeView",filepath)


#TODO Better
if input("Would you like to plot Q [True]?"):
    environment = exp.getMDP(1)
    for i in expVar.iterationLoaded:
        path = experimentPath + "/0Q/0/0_0_" + str(i) + ".npy"
        l = []
        xs = np.linspace(-environment.max_pos, environment.max_pos, 60)
        us = np.linspace(-environment.max_action, environment.max_action, 50)

        tabular_Q = np.load(path)

        # print(tabular_Q.shape)
        # print(tabular_Q)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(tabular_Q[:, 0], tabular_Q[:, 1], tabular_Q[:, 2])
        filepath = "plot/" + experimentPath + "/" + "QRegressor.jpg"
        directory = os.path.dirname(filepath)
        if not os.path.isdir(directory): os.makedirs(directory)
        plt.savefig(filepath)
        plt.show()
