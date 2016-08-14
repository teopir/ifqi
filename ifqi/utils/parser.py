from __future__ import print_function
import os
import csv
import numpy as np

def parseReLeDataset(path, nEpisodes=None):
    """
    Function to parse rele dataset and create
    a sars matrix with tuples with:
    state, action, reward, next state, absStateFlag
    
    """
    fileName = os.path.realpath(path)

    print("Loading dataset...")
    dataList = list()
    with open(fileName, 'r') as f:
        episodesCounter = 0
        csvReader = csv.reader(f, delimiter=',')
        first = True
        for row in csvReader:
            if first:
                first = False
                stateDim = int(row[0])
                actionDim = int(row[1])
                rewardDim = int(row[2])
            else:
                if len(row) == stateDim + 2:
                    currentRow = row + [0]*(actionDim+rewardDim)
                    dataList.append(currentRow)
                    episodesCounter += 1
                else:
                    dataList.append(row)
            
            if nEpisodes is not None and episodesCounter == nEpisodes:
                break

    print("Dataset loaded")

    data = np.array(dataList, dtype='float32')

    statepos = 2
    actionpos = statepos + stateDim
    rewardpos = actionpos + actionDim

    idxs = np.argwhere(data[:, 0] != 1).ravel()
    states = data[idxs, statepos:actionpos].reshape(-1, stateDim)
    actions = data[idxs, actionpos:rewardpos].reshape(-1, actionDim)
    rewards = data[idxs, rewardpos:rewardpos + rewardDim].reshape(-1, rewardDim)
    nextStates = data[idxs + 1, statepos:actionpos].reshape(-1, stateDim)
    absorbingStates = data[idxs + 1, 1].reshape(-1,1)

    sars = np.concatenate((states, actions, rewards, nextStates, absorbingStates), axis=1)

    return sars, stateDim, actionDim, rewardDim

"""
def parsejson(path):

    with open(path, 'r') as infile:
        jsondata = json.load(infile)

        dataList = jsondata['data']
        flatlist = []
        for el in dataList:
            flatlist += el
        data = np.array(flatlist, dtype='float32')

        return data, jsondata['statedim'], jsondata['actiondim'], jsondata['rewarddim']
"""