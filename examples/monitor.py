"""
This script will help you to visualize how an experiment is evolving
"""
import os
import time
import numpy as np
from examples.variableLoadSave import  ExperimentVariables
import pylab as plb
import matplotlib.pyplot as plt

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

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

expVar = ExperimentVariables(experimentPath)

fig1 = plt.figure(1)
ax = fig1.add_subplot(1, 1, 1)
h = ax.plot([], [], 'ro-')
plt.ion()  # turns on interactive mode
plt.show()

subplots = {}
figplots = {}
min_x = np.infty
min_y = np.infty
max_x = -np.infty
max_y = -np.infty

while True:
    try:
        ret = expVar.getOverallSituation("score")
        for key in ret:
            if not key in subplots:
                temp = fig1.add_subplot(1,1,1)
                figplots[key] = temp
                subplots[key] = temp.plot(ret[key][0],ret[key][1])
            subplots[key][0].set_data(ret[key][0],ret[key][1])

            if min(ret[key][1]) < min_y:
                min_y = min(ret[key][1])
            if min(ret[key][0]) < min_x:
                min_x = min(ret[key][0])
            if max(ret[key][1]) > max_y:
                max_y = max(ret[key][1])
            if max(ret[key][0]) > max_x:
                max_x = max(ret[key][0])

            figplots[key].figure.canvas.draw()

        boundaries = max(abs(min_y*0.1),abs(max_y*0.1))
        plt.ylim(min_y - boundaries, max_y + boundaries)
        plt.xlim(0, max_x)
    except:
        pass
    clear()
    time.sleep(1)
