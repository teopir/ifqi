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
import re

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

data = {}
for key in get:
    #Ask if he/she whants to add the line to the plot
    data[key] = {}
    data[key]["name"] = str(exp.getModelName(key[0])) + "_" + str(key[1])
    data[key]["iteration"] = get[key][0]
    data[key]["mean"] = get[key][1]
    data[key]["std"] = get[key][2]

            
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
def multiOneDimPlot(data,xlabel,ylabel,title,path):
    color = ["r","g","b","k"]
    imgName = os.path.realpath(path)
    i=0
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
        
        if(len(std)!=0):
            plt.errorbar(dict_["iteration"],mean,color=color[i],yerr=std,lw=1,label=name)
        hand.append(plt.plot(dict_["iteration"],mean,color=color[i],lw=1,label=name)[0])

        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        
        i+=1
        i%=len(color)
    
    plt.legend(handles=hand)
    plt.ylim(min_ - abs(min_) * 0.1 , max_ + 0.1 * abs(max_))   
    plt.savefig(imgName)
    plt.show()

multiOneDimPlot(data,"iterations","score","ciao","prova.jpg")

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
