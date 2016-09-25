import os
import sys
import csv
import numpy as np

sys.path.append(os.path.abspath('../'))


def loadReLeDataset(path, nEpisodes=None):
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
                    currentRow = row + [0] * (actionDim + rewardDim)
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
    rewards = data[idxs, rewardpos:rewardpos + rewardDim].reshape(-1,
                                                                  rewardDim)
    nextStates = data[idxs + 1, statepos:actionpos].reshape(-1, stateDim)
    absorbingStates = data[idxs + 1, 1].reshape(-1, 1)

    sars = np.concatenate((states,
                           actions,
                           rewards,
                           nextStates,
                           absorbingStates), axis=1)

    return sars, stateDim, actionDim, rewardDim


def loadIfqiDataset(path, nEpisodes=None):
    """
    This function loads the dataset with the given number of episodes
    from a .npy file saved by the main function of this module.

    Args:
        - path (string): the path of the file containing the dataset
        - nEpisodes (int): the number of episodes to consider inside the
                           dataset

    Returns:
        - the loaded dataset

    """
    assert nEpisodes > 0, "Number of episodes to compute must be greater than \
                           zero."
    data = np.load(path)
    episodeStarts = np.argwhere(data[:, 0] == 1).ravel()

    if nEpisodes is not None:
        assert np.size(episodeStarts) >= nEpisodes

        return data[:episodeStarts[nEpisodes - 1] + 1, 1:]
    return data[:, 1:]


from experiment import Experiment


def collect(mdp, nEpisodes):
    """
    Collection of the dataset using the given environment and number
    of episodes.

    Args:
        - mdp (object): the environment to consider
        - nEpisodes (int): the number of episodes

    Returns:
        - the dataset

    """
    data = mdp.collectSamples()

    for i in xrange(nEpisodes - 1):
        data = np.append(data, mdp.collectSamples(), axis=0)

    return data

if __name__ == '__main__':
    mdpName = 'LQG1D'
    fileName = 'lqg'
    nEpisodes = 1000

    exp = Experiment()
    exp.config['mdp'] = dict()
    exp.config['mdp']['mdpName'] = mdpName
    mdp = exp.getMDP()
    data = collect(mdp, nEpisodes)

    np.save('../../dataset/' + mdpName + '/' + fileName, data)
