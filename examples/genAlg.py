"""
Genetic algorithm for choosing the best MLP
"""

from __future__ import print_function

import os
import sys
import time
import json

from ifqi.experiment import Experiment
import subprocess
from subprocess import Popen, PIPE

from random import shuffle
import argparse

import numpy as np
import time
import random


#----------------------------------------------------------------------------------------------
# Argument parser
#----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description='Execution of one experiment thread provided a configuration file and\n\t A regressor (index)\n\t Size of dataset (index)\n\t Dataset (index)')

parser.add_argument("envName", type=str, help="Provide the name of the environment")
parser.add_argument("nPop", type=int, help="Provide the size of the population")
parser.add_argument("nCore", type=int, help="Provides the number of core to use")
parser.add_argument("outFile", type=str, help="Provides the name of the file")
parser.add_argument("nDataset", type=int, help="Provides the number dataset to use",nargs="?", default=1)
parser.add_argument("nEval", type=int, help="Provides the number of evaluation to use",nargs="?", default=1)
parser.add_argument("nIterMin", type=int, help="Provides the minimum number of iterations",nargs="?", default=1)
parser.add_argument("nIterMax", type=int, help="Provides the maximum number of iterations",nargs="?", default=1)
parser.add_argument("sSeed", type=int, help="Provides the starting seed",nargs="?", default=0)
parser.add_argument("-cont", action='store_const',
                    const=True, default=False,
                    help='Loads the epoch file if it exists')
parser.add_argument("-extra", action='store_const',
                    const=True, default=False,
                    help='Loads the epoch file if it exists')

args = parser.parse_args()

configFile = args.envName
nCore = args.nCore
nPop = args.nPop
outFile = args.outFile
nDataset = args.nDataset
nEval = args.nEval
nIterMin = args.nIterMin
nIterMax = args.nIterMax
sSeed = args.sSeed
cont = args.cont
extraTree = args.extra

#----------------------------------------------------------------------------------------------
# Genetic core
#----------------------------------------------------------------------------------------------

if not extraTree:
    """ranges = {
    "n_epochs": (500,501),
    "activation": (0,3),
    "batch_size": (2000,2001),
    "n_neurons": (10,10),
    "n_layers": (2,3),
    "delta_minA": (0,10),
    "delta_minB": (3,16),
    "patienceA": (0,10),
    "patienceB": (0,3)
}"""

    ranges = [(5000,5001),(0,3),(2000,2001),(10,11),(2,3),(10,100),(1,8),(0,10),(0,3)]#,(0,2), (0,2)]

    mutations = [0., 0.15, 0., 0., 0.,0.15,0.15,0.15,0.15]#, 0.4, 0.4]
else:
    """ranges = {
        "n_estimators": (1,50),
        "min_sample_split": (1,10),
        "min_sample_leaf": (1,10),
        "min_weight_fraction_leaf": (1,1000),
        "bootstrap": (0, 2),
        "max_depth": (-200, 1000),
    }"""

    ranges = [(1, 50), (1, 10), (1, 10), (0, 1000), (0, 2), (-200,1000)]

    mutations = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

p_mut = 0.8
if os.path.isfile("DNA.json"):
    with open("DNA.json") as dna:
        data = json.load(dna)
        ranges = data["ranges"]
        mutations = data["mutations"]
        p_mut = data["p_mut"]


def generates():
    global  ranges
    return map(lambda(x): np.random.randint(*x), ranges)

def mutate(dna):
    global ranges, p_mut
    if np.random.rand() < p_mut:
        muting_dna = zip(dna, mutations, ranges)
        dna = map(lambda(x): x[0] if np.random.rand() < x[1] else np.random.randint(*x[2]), muting_dna )
    return dna

def combine(dna1, dna2):
    dnas = zip(dna1,dna2)
    dnas = map(lambda(x): (min(x),max(x)+1), dnas)
    new_dna =  map(lambda(x): np.random.randint(*x), dnas)
    return mutate(new_dna)

population = None
evaluation = [-np.inf]

def newPopulation():
    global population, evaluation, outFile
    nPop = len(population)

    indxs = [i[0] for i in sorted(enumerate(evaluation), key=lambda x: x[1])]
    indxs.reverse()

    population = [population[i] for i in indxs]
    evaluation = [evaluation[i] for i in indxs]

    popeval = zip(population, evaluation)
    print("-"*20)
    print("BestOne:", popeval[0])
    x = open(outFile+".txt","a")
    x.write(str(popeval[0]) +"\n")
    x.close()
    print("WorstOne:", popeval[-1])
    print("-"*20)

    elite = population[:nPop / 5]
    nElite = len(elite)

    bestPop = population[:nPop / 2]
    nbPop = len(bestPop)
    population = elite + [combine(bestPop[np.random.randint(nbPop)],bestPop[np.random.randint(nbPop)]) for i in range(nPop-nElite)]

def evaluate():
    global population,evaluation, nCore, configFile, nIter, sSeed

    myPath = os.path.realpath(__file__)
    myPath = os.path.dirname(myPath)
    if extraTree:
        myPath += "/genExtraThread.py"
    else:
        myPath += "/genThread.py"

    processes = set()
    population_process = []
    # command should be a list
    try:

        for individual in population:
            population_process.append([])
            for ds in range(nDataset):
                p = subprocess.Popen(["python", myPath, configFile] + [str(x) for x in individual] + [str(ds + sSeed),str(nIter), str(nEval)], stdout=PIPE)
                population_process[-1].append(p)
                processes.add(p)
                while len(processes) >= nCore:
                    time.sleep(0.5)
                    processes.difference_update([p for p in processes if p.poll() is not None])

        for p in processes:
            if p.poll() is None:
                p.wait()

    except KeyboardInterrupt:
        print("Keyboard interrupt catch")
        for p in processes:
            p.kill()
            exit()
    #print("x.communicate",[x.communicate() for x in population_process[0]])
    clean = lambda(x): float(str(x.communicate()[0]).split()[-1])
    evaluation = [np.mean(map(clean,p)) for p in population_process]
    print("evaluation", evaluation)


population = map(lambda(x): generates(), [None]*nPop)

if os.path.isfile(outFile + ".epoch"):
    if cont:
        with open(outFile + ".epoch","r") as data_file:
            data = json.load(data_file)
        loadedPop = data["population"]
        if len(population) < len(loadedPop):
            loadedPop = loadedPop[:len(population)]
        for (individual,i) in zip(loadedPop, range(len(loadedPop))):
            population[i] = individual


print(population)
nIter = nIterMin
while True:
    evaluate()
    newPopulation()
    with open(outFile + ".epoch","w") as data_file:
        print (population)
        json.dump( {"population":population},data_file)
    if nIter < nIterMax:
        nIter += 1