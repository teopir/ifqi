from __future__ import print_function

import os
import sys
sys.path.append(os.path.abspath('../'))

import unittest
import numpy as np
from ifqi.fqi.FQI import FQI
from ifqi.models.mlp import MLP
from ifqi.preprocessors.invPendulumPreprocessor import InvertedPendulumPreprocessor
import ifqi.utils.parser as parser
from sklearn.ensemble import ExtraTreesRegressor


if __name__ == '__main__':

    data, sdim, adim, rdim = parser.parseReLeDataset('../dataset/episodicPendulum.txt')
    assert(sdim == 2)
    assert(adim == 1)
    assert(rdim == 1)

    rewardpos = sdim + adim
    indicies = np.delete(np.arange(data.shape[1]), rewardpos)

    # select state, action, nextstate, absorbin
    sast = data[:, indicies]

    prep = InvertedPendulumPreprocessor()
    sast[:,:3] = prep.preprocess(sast[:,:3])

    # select reward
    r = data[:, rewardpos]

    estimator = 'mlp'
    if estimator == 'extra':
        alg = ExtraTreesRegressor(n_estimators=50, criterion='mse',
                                         min_samples_split=2, min_samples_leaf=1)
        fit_params = dict()
    elif estimator == 'mlp':
        alg = MLP(n_input=sdim+adim, n_output=1, hidden_neurons=5, h_layer=2,
                  optimizer='rmsprop', act_function="sigmoid").getModel()
        fit_params = {'nb_epoch':12, 'batch_size':50, 'verbose':1}
        # it is equivalente to call
        #fqi.fit(sast,r,nb_epoch=12,batch_size=50, verbose=1)
    else:
        raise ValueError('Unknown estimator type.')

    actions = (np.arange(10) - 1).tolist()
    fqi = FQI(estimator=alg,
              stateDim=sdim, actionDim=adim,
              discrete_actions=actions,
              gamma=0.95, horizon=10, verbose=1)
    #fqi.fit(sast, r, **fit_params)

    for t in range(20):
        fqi.partial_fit(sast, r, **fit_params)
        mod = fqi.estimator
        ## test on the simulator