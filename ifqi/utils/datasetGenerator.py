"""
This file create, manage, and save on disk a dataset
"""
import numpy as np
import csv
import os
import ifqi.evaluation.evaluation as evaluate


class DatasetGenerator:

    def __init__(self, environment):
        self.environment = environment
        self._stateDim = self.environment.observation_space.shape[0]
        #TODO: have to generalize with more actions
        self.data = np.zeros((0, 3 + self._stateDim * 2 + 1))


    def save(self, fileName):
        """
        Let's save the dataset on disc
        :param fileName:  Name of the file
        :return: Nothing
        """
        np.save(fileName, self.data)
        
    def load(self,fileName):
        # type: (str) -> np.ndarray
        """
        Load dataset from file
        :param fileName: file name
        :return: numpy matrix
        """
        self.data = np.load(fileName)
        
    def loadReLeDataset(self, path, nEpisodes=None):
        """
        Function to parse rele dataset and create
        a sars matrix with tuples with:
        state, action, reward, next state, absStateFlag
    
        """
        raise Exception("Not implemented yet")

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
                    assert stateDim == self.environment.observation_space.shape[0], "Dimension of the state must be the same"
                    
                else:
                    if len(row) == stateDim + 2:
                        currentRow = row + [0] * (actionDim + rewardDim)
                        dataList.append(currentRow)
                        episodesCounter += 1
                    else:
                        dataList.append(row)
    
                if nEpisodes is not None and episodesCounter == nEpisodes:
                    break
    
        data = np.array(dataList, dtype='float32')
    
        statepos = 2
        actionpos = statepos + stateDim
        rewardpos = actionpos + actionDim
    
        idxs = np.argwhere(data[:, 0] != 1).ravel()
        states = data[idxs, statepos:actionpos].reshape(-1, stateDim)
        actions = data[idxs, actionpos:rewardpos].reshape(-1, actionDim)
        rewards = data[idxs, rewardpos:rewardpos + rewardDim].reshape(-1,
                                                                      rewardDim)
        nextStates = data[idxs + 1, statepos:actionpos].reshape(-1, stateDim)
        absorbingStates = data[idxs + 1, 1].reshape(-1, 1)
        #TODO:
        self.data = None #np.concatenate()

    def loadAppend(self, fileName):
        """
        Let's load the dataset from the disk without throwing away the actual dataset
        :param fileName: name of the file
        :return: np.ndarray
        """
        tempData = np.load(fileName)
        self.data = np.concatenate((self.data, tempData))

    def generateAppend(self, policy=None, n_episodes=100):
        """
        Here we generate some data given a pilicy and a number of dataset
        :param policy: policy (if None then it will be random)
        :param n_episodes: natural positive number of episode
        :return: nothing
        """
        for _ in xrange(n_episodes):
            tempData = evaluate.collectEpisode(self.environment, policy)
            self.data = np.concatenate((self.data, tempData), axis=0)

    def generate(self, policy=None, n_episodes=100):
        """
        Here we generate some data given a pilicy and a number of dataset
        :param policy: policy (if None then it will be random)
        :param n_episodes: natural positive number of episode
        :return: nothing
        """
        self.reset()
        for _ in xrange(n_episodes):
            tempData = evaluate.collectEpisode(self.environment, policy)
            self.data = np.concatenate((self.data, tempData), axis=0)

    def reset(self):
        """
        Here the dataset is reset
        :return:
        """
        self.data = np.zeros((0, 3 + self._stateDim * 2 + 1))

    @property
    def action(self):
        """
        column of the action
        :return: ndarray
        """
        return self.data[:, 1+self._stateDim]

    @property
    def state(self):
        """
        matrix of the state
        :return:
        """
        return self.data[:, 1:1 + self._stateDim]

    @property
    def nextState(self):
        """
        matrix of the next state
        :return:
        """
        return self.data[:, self._stateDim + 3: 2 * self._stateDim + 3]

    @property
    def absorbing(self):
        """
        is the state absorbing
        :return:
        """
        return self.data[:, -1]

    @property
    def endEpisode(self):
        """
        is it the end of the episode
        :return:
        """
        return self.data[:, 0]

    @property
    def reward(self):
        """
        reward column
        :return:
        """
        return self.data[:, self._stateDim + 2]

    @property
    def sastr(self):
        """
        state, action, nextstate, terminate=True, reward
        :return: (sast, r)
        """
        return (np.concatenate((self.state, np.matrix(self.action).T, self.nextState, np.matrix(self.absorbing).T), axis=1), self.reward)
