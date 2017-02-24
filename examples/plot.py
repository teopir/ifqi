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
from variableLoadSave import ExperimentVariables
import argparse
import json
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import re
from matplotlib.backends.backend_pdf import PdfPages

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
parser = argparse.ArgumentParser(
        description="""Welcome in plot.py.
        """)
parser.add_argument("folder", type=str, help="Provide the folder of the experiment")
parser.add_argument("variable", type=str, help="The name of the variable to plot")
parser.add_argument("-s", "--size", help="size", action="store_true")

parser.add_argument("-tl", "--topleft", help="topleft legend", action="store_true")
parser.add_argument("-lx", "--logx", help="topleft legend", action="store_true")
parser.add_argument("-m", "--mlp", help="mlp", action="store_true")

descr =""
args = parser.parse_args()
folder = args.folder
show_sizes = args.size
variable = args.variable
top_left = args.topleft
logx = args.logx
mlp = args.mlp

sizeTitle = "Extra Trees"
if mlp:
    sizeTitle = "Neural Network"
font = {
        'size'   : 22,
        'style' : "normal"}

font_label = {
        'size'   : 27,
        'style' : "normal"}
mpl.rc('font', **font)

def multiOneDimPlot(data, xlabel, ylabel, title, path):
    global font_label, top_left, logx
    color = ["r", "g", "b", "k"]
    dashes = [":", "--", "-."]
    imgName = os.path.realpath(path)

    i = 0
    hand = []
    min_x = np.inf
    min_ = np.inf
    max_ = -np.inf
    max_x = -np.inf
    sort_keys = sorted(data)
    for dic in sort_keys:
        dict_ = data[dic]
        name = dict_["name"]
        mean = dict_["mean"]
        conf = dict_["conf"]


        s = dict_["iteration"]
        indxs = sorted(range(len(s)), key=lambda k: s[k])

        dict_["iteration"] = sorted(dict_["iteration"])
        mean = [dict_["mean"][jj] for jj in indxs]
        conf = [dict_["conf"][jj] for jj in indxs]

        if max(dict_["iteration"]) > max_x:
            max_x = max(dict_["iteration"])
        if min(dict_["iteration"]) < min_x:
            min_x = min(dict_["iteration"])
        if min(mean) < min_:
            min_ = min(mean)
        if max(mean) > max_:
            max_ = max(mean)

        print "len", len(hand)
        if (len(conf) != 0):
            plt.errorbar(dict_["iteration"], mean, color=color[i], yerr=conf, lw=2, label=name, ls=dashes[i])
        if logx:
            temp = plt.semilogx(dict_["iteration"], mean, color=color[i], lw=2, label=name, ls=dashes[i])
        else:
            temp = plt.plot(dict_["iteration"], mean, color=color[i], lw=2, label=name, ls=dashes[i])

        hand.append(temp[0])

        plt.ylabel(ylabel,  fontdict=font_label)
        plt.xlabel(xlabel)
        plt.title(title)

        i += 1
        i %= len(color)

    loc = 4
    if top_left:
        loc=0
    plt.legend(handles=hand, loc=loc)
    boundaries = max(abs(min_) * 0.1, 0.1 * abs(max_))
    plt.ylim(min_ - boundaries, max_ + boundaries)
    plt.xlim(min_x * 0.95, max_x * 1.05 )
    print "save PDF"
    pp = PdfPages(imgName.split('.')[0] + ".pdf")
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()
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


experimentPath = folder
if descr=="":
    descr = folder + ".json"
jsonFile = descr
exp = Experiment(configFile=jsonFile)
expVar = ExperimentVariables(experimentPath)


get = expVar.getOverallSituation(variable)
nLines = len(get)
print(str(nLines) + " lines are found.")
if not(show_sizes):
    for s in expVar.sizeLoaded:
        data = {}
        for reg in expVar.regressorLoaded:
            key = (reg, s)
            data[key] = {}
            size = exp.config['experimentSetting']['sizes'][s]
            data[key]["name"] = str(exp.getModelName(reg)) #+ "_" + str(size)
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
        if variable=="score":
            lab = "$J^{\pi_N}$"
        elif variable=="steps":
            lab = "#steps"
        else:
            lab = variable
        multiOneDimPlot(data,"#iterations", lab, sizeTitle, filepath)

if show_sizes:
    get = expVar.getSizeLines(variable)
    max_iter = sorted(expVar.iterationLoaded)[-1]
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
    multiOneDimPlot(data,"#episodes","$J^{\pi_{" + str(max_iter) + "}}$" if variable=="score" else variable, sizeTitle,filepath)


#TODO Better
if False and input("Would you like to plot Q [True]?"):
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
