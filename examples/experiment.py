from __future__ import print_function
import os
import sys

import numpy as np

sys.path.append(os.path.abspath('../'))

from ifqi.experiment import Experiment
from ifqi.fqi.FQI import FQI
from ifqi.utils import parser

# Python 2 and 3: forward-compatible
# from builtins import range


"""
This script can be used to launch experiments using settings
provided in a json configuration file.
The script computes and save the performance of the algorithm
and model in the selected environment averaging on different
experiments and different datasets. While the loop over experiments
is likely to be used for every test, the loop over dataset is not.
Indeed one could prefer to iterate over different number of FQI
steps and so on.

"""
if __name__ == '__main__':
    configFile = sys.argv[1]

    exp = Experiment(configFile)

    if 'MLP' in exp.config['model']['modelName']:
        fitParams = {'nb_epoch': exp.config['supervisedAlgorithm']
                                           ['nEpochs'],
                     'batch_size': exp.config['supervisedAlgorithm']
                                             ['batchSize'],
                     'validation_split': exp.config['supervisedAlgorithm']
                                                   ['validationSplit'],
                     'verbose': exp.config['supervisedAlgorithm']
                                          ['verbosity']
                     }
    else:
        fitParams = dict()

    score = np.zeros((exp.config['experimentSetting']['nExperiments'],
                      exp.config['experimentSetting']['nDatasets']))

    for d in range(exp.config['experimentSetting']['nDatasets']):
        print('Dataset: ' + str(d))
        loadPath = exp.config['experimentSetting']['loadPath']
        if loadPath.endswith('npy'):
            data = np.load('../dataset/' + loadPath)
            stateDim, actionDim = exp.mdp.stateDim, exp.mdp.actionDim
            rewardDim = 1
        else:
            data, stateDim, actionDim, rewardDim = parser.parseReLeDataset(
                path='../dataset/' + loadPath,
                nEpisodes=(d + 1) * exp.config['experimentSetting']
                                              ['datasetSizeStep']
                )
            assert(stateDim == exp.mdp.stateDim)
            assert(actionDim == exp.mdp.actionDim)
            assert(rewardDim == 1)

        rewardpos = stateDim + actionDim
        indicies = np.delete(np.arange(data.shape[1]), rewardpos)
        sast = data[:, indicies]
        r = data[:, rewardpos]

        for e in range(exp.config['experimentSetting']['nExperiments']):
            print('Experiment: ' + str(e))

            exp.loadModel()

            if 'features' in exp.config['model']:
                features = exp.config['model']['features']
            else:
                features = None

            fqi = FQI(estimator=exp.model,
                      stateDim=stateDim,
                      actionDim=actionDim,
                      discreteActions=range(exp.mdp.nActions),
                      gamma=exp.config['rlAlgorithm']['gamma'],
                      horizon=exp.config['rlAlgorithm']['horizon'],
                      verbose=exp.config['rlAlgorithm']['verbosity'],
                      features=features,
                      scaled=exp.config['rlAlgorithm']['scaled'])

            fqi.partialFit(sast, r, **fitParams)
            for t in range(1, exp.config['rlAlgorithm']['nIterations']):
                fqi.partialFit(None, None, **fitParams)

            score[e, d] = exp.mdp.evaluate(fqi)[0]

    np.save(exp.config['experimentSetting']['savePath'], score)
