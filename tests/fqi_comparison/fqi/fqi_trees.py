# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:30:53 2016

@author: samuele
"""
from __future__ import print_function
import csv
import os
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from copy import copy
import time

class FQI():
    def __init__(self, nActions, gamma=0.9,
                 nIterations=10, nb_epoch=1, batch_size=100,
                 val_split=0, patience=500):
        self.nActions = nActions
        self.gamma = gamma
        self.nIterations = nIterations
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.models = None
        self.stateDim = None
        self.val_split = val_split
        self.patience = patience
    
    """
    Function to parse rele dataset and create
    a sars matrix with tuples with:
    state, action, reward, next state, absStateFlag
    """
    def parseDataset(self, dataPath):
        fileName = os.path.realpath(dataPath)
        
        print("Reading dataset ...")
        dataList = list()
        with open(fileName, 'r') as f:
            csvReader = csv.reader(f, delimiter=',')
            first = True
            for row in csvReader:
                if first:
                    first = False
                    self.stateDim = int(row[0])
                else:
                    if len(row) > self.stateDim + 2:
                        dataList.append(row)
                    else:
                        currentRow = row
                        currentRow.append('0')
                        currentRow.append('0')
                        dataList.append(currentRow)
        print("Data read")
        
        data = np.array(dataList, dtype='float32')

        idxs = np.argwhere(data[:, 0] != 1).ravel()
        states = data[idxs, 2:self.stateDim + 2]
        actions = data[idxs, self.stateDim + 2]
        rewards = data[idxs, self.stateDim + 3]
        nextStates = data[idxs + 1, 2:self.stateDim + 2]
        absorbingStates = data[idxs + 1, 1]
            
        sars = np.concatenate((np.matrix(states), np.matrix(actions).T,
                               np.matrix(rewards).T, np.matrix(nextStates),
                               np.matrix(absorbingStates).T), axis=1)

        return sars

    def first_run(self, sars, preprocessor=None, save=False, fileName=""):
        print("Starting FQI ...")
            
        actionIdx = self.stateDim
        rewardIdx = self.stateDim + 1
        absorbingStateIdx = 2 * self.stateDim + 2
        
        self.model = ExtraTreesRegressor(n_estimators=50, criterion='mse', 
                                         min_samples_split=2, min_samples_leaf=1,
                                         random_state = 54)

        # FQI
        
        start = time.time()
        rawStates = copy(sars[:, :self.stateDim])
        actions = copy(sars[:, actionIdx])
        trainX = preprocessor.preprocess(np.concatenate((rawStates, actions), axis=1))
        trainY = copy(np.array(sars[:, rewardIdx].T)[0])
        
        self.model.fit(trainX, trainY)
        
        self.preprocessor=preprocessor
        

        print("FQI1: " + str(int(time.time() -start)) + "s")
            
        self.sars = sars
        self.actionIdx = actionIdx
        self.absorbingStateIdx = absorbingStateIdx
        self.rewardIdx = rewardIdx
            
        self.i = 1

        return trainX, trainY
            
    def run(self):
        self.i += 1
        preprocessor = self.preprocessor
        sars = self.sars
        actionIdx = self.actionIdx
        absorbingStateIdx = self.absorbingStateIdx
        rewardIdx = self.rewardIdx
        
        start = time.time()
        rawTrainX = copy(sars[:, :actionIdx + 1])
        trainX = preprocessor.preprocess(rawTrainX)
            
        Q = np.zeros((sars.shape[0], self.nActions))
        for k in range(self.nActions):
            rawNextStates = copy(sars[:, self.stateDim + 2:2 * self.stateDim + 2])
            currentAction = np.ones((sars.shape[0], 1)) * k
            sample = preprocessor.preprocess(np.concatenate(
                                                (rawNextStates, np.matrix(currentAction)),
                                                 axis=1))
            currentOutput = self.model.predict(sample)
            
            Q[:, k] = currentOutput.ravel()
            Q[:, k] = Q[:, k] * (1 - np.asarray(sars[:, absorbingStateIdx]).reshape(-1))
            
        trainY = sars[:, rewardIdx] + np.matrix(self.gamma * np.max(Q, axis=1)).T

        flat_trainY = np.array(trainY.T)[0]
            
        self.model.fit(trainX, flat_trainY)

        print("FQI" + str(self.i) + ": " + str(int(time.time() -start)) + "s")

        return trainX, flat_trainY
        