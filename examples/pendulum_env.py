from __future__ import print_function

import os
import sys
sys.path.append(os.path.abspath('../'))

import unittest
import numpy as np
from ifqi.fqi.FQI import FQI
from ifqi.models.mlp import MLP
from ifqi.models.incr import WideRegressor
from ifqi.preprocessors.invPendulumPreprocessor import InvertedPendulumPreprocessor
from ifqi.envs.invertedPendulum import InvPendulum
import ifqi.utils.parser as parser
from sklearn.ensemble import ExtraTreesRegressor

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def runEpisode(myfqi, environment, gamma):   # (nstep, J, success)
    J = 0
    t=0
    test_succesful = 0
    rh = []
    while(not environment.isAbsorbing()):
        state = environment.getState()
        action, _ = myfqi.predict(np.array(state))
        r = environment.step(action)
        J += gamma**t * r
        if(t>3000):
            print('COOL! you done it!')
            test_succesful = 1
            break
        t+=1
        rh += [r]
    return (t, J, test_succesful), rh

if __name__ == '__main__':

    #data, sdim, adim, rdim = parser.parseReLeDataset('../dataset/episodicPendulum.txt')
    data, sdim, adim, rdim = parser.parseReLeDataset('../dataset/pendulum_data/F/data0.log')
    assert(sdim == 2)
    assert(adim == 1)
    assert(rdim == 1)

    rewardpos = sdim + adim
    indicies = np.delete(np.arange(data.shape[1]), rewardpos)

    # select state, action, nextstate, absorbin
    sast = data[:, indicies]

    #prep = InvertedPendulumPreprocessor()
    #sast[:,:3] = prep.preprocess(sast[:,:3])

    # select reward
    r = data[:, rewardpos]

    estimator = 'mlp'
    niteration = 20
    if estimator == 'extra':
        alg = ExtraTreesRegressor(n_estimators=50, criterion='mse',
                                         min_samples_split=2, min_samples_leaf=1)
        fit_params = dict()
    elif estimator == 'mlp':
        alg = MLP(n_input=sdim+adim, n_output=1, hidden_neurons=5, h_layer=2,
                  optimizer='rmsprop', act_function="sigmoid")
        fit_params = {'nb_epoch':300, 'batch_size':50, 'verbose':0}
        # it is equivalente to call
        #fqi.fit(sast,r,nb_epoch=12,batch_size=50, verbose=1)
    elif estimator == "wide":
        alg = WideRegressor(n_input=sdim+adim, n_output=1, hidden_neurons=[5]*niteration,
                            n_h_layer_beginning=1, optimizer='rmsprop', act_function=["sigmoid"]*niteration)
        fit_params = {'nb_epoch':300, 'batch_size':50, 'verbose':0}
    else:
        raise ValueError('Unknown estimator type.')

    actions = (np.arange(3)).tolist()
    fqi = FQI(estimator=alg,
              stateDim=sdim, actionDim=adim,
              discrete_actions=actions,
              gamma=0.99, horizon=10, verbose=1, scaled=True)
    #fqi.fit(sast, r, **fit_params)

    environment = InvPendulum()


    fqi.partial_fit(sast, r, **fit_params)
    for t in range(1, niteration):
        fqi.partial_fit(None, None, **fit_params)
        mod = fqi.estimator
        ## test on the simulator
        print('Simulate on environment')
        environment.reset()
        tupla, rhistory = runEpisode(fqi, environment, 0.95)
        #plt.scatter(np.arange(len(rhistory)), np.array(rhistory))
        #plt.show()