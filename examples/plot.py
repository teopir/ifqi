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
        std = dict_["std"]

        if min(mean) < min_:
            min_ = min(mean)
        if max(mean) > max_:
            max_ = max(mean)

        if (len(std) != 0):
            plt.errorbar(dict_["iteration"], mean, color=color[i], yerr=std, lw=1, label=name)
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
variable = ask("Which variable would you like to consider? ", "Please insert a string for the name", str)
exp = Experiment(configFile=jsonFile)

expVar = ExperimentVariables(experimentPath)
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
        data[key]["mean"] = get[key][1]
        data[key]["std"] = get[key][2]
    filepath = "plot/" + experimentPath + "/" + "size" + str(size)+".jpg"
    directory =os.path.dirname(filepath)
    if not os.path.isdir(directory): os.makedirs(directory)
    multiOneDimPlot(data,"iterations","score","size=" + str(size),filepath)

get = expVar.getSizeLines("score")
data = {}
for reg in expVar.regressorLoaded:
    key=reg
    data[key] = {}
    data[key]["name"] = str(exp.getModelName(reg))
    data[key]["iteration"] = [exp.config['experimentSetting']['sizes'][x] for x in get[key][0]]
    data[key]["mean"] = get[key][1]
    data[key]["std"] = get[key][2]

filepath = "plot/" + experimentPath + "/" + "sizeview.jpg"
directory =os.path.dirname(filepath)
if not os.path.isdir(directory): os.makedirs(directory)
multiOneDimPlot(data,"sizes","score","SizeView",filepath)


if input("Would you like to plot Q [True]?"):
    environment = exp.getMDP(1)
    path = raw_input("path of the file")
    tabular_Q = np.load(path)
    l = []
    xs = np.linspace(-environment.max_pos, environment.max_pos, 60)
    us = np.linspace(-environment.max_action, environment.max_action, 50)


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

path = "prova.jpg"
if input("Would you like to save [True]?"):
    plot_name = raw_input("Plot title: ")
    description = raw_input("Plot description: ")
    path = "plot/" + experimentPath + "/" + plot_name + ".jpg"
    directory =os.path.dirname(path)
    if not os.path.isdir(directory): os.makedirs(directory)


    myExp["images"].append({"title":plot_name,
    "dir":path,
    "description":description
    }
    )
    with open("results/diary.json", 'w') as fp:
        json.dump(diary,fp)

for varname in varnames:
    for size in sizes:
        multiOneDimPlot(data_plot[varname][size],"iteration",varname,"variable: "+varname + " dataset size: " + str(size),path)
