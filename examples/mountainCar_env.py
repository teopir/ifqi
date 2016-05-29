from __future__ import print_function

import os
import sys
sys.path.append(os.path.abspath('../'))

import unittest
import numpy as np
from ifqi.fqi.FQI import FQI
from ifqi.models.mlp import MLP
from ifqi.models.incr import MergedRegressor, WideRegressor
from ifqi.preprocessors.mountainCarPreprocessor import MountainCarPreprocessor
from ifqi.envs.mountainCar import MountainCar
import ifqi.utils.parser as parser
from sklearn.ensemble import ExtraTreesRegressor

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def runEpisode(myfqi, environment, gamma):   # (nstep, J, success)
    J = 0
    t = 0
    test_succesful = 0
    rh = []
    while(t < 500 and not environment.isAbsorbing()):
        state = environment.getState()
        action, _ = myfqi.predict(np.array(state))
        position, velocity, r = environment.step(action)
        J += gamma ** t * r
        t += 1
        rh += [r]
        
        if r == 1:
            print('Goal reached')
            test_succesful = 1

    return (t, J, test_succesful), rh

if __name__ == '__main__':

    data, sdim, adim, rdim = parser.parseReLeDataset('../dataset/mountainCar.txt')
    assert(sdim == 2)
    assert(adim == 1)
    assert(rdim == 1)

    rewardpos = sdim + adim
    indicies = np.delete(np.arange(data.shape[1]), rewardpos)

    # select state, action, nextstate, absorbin
    sast = data[:, indicies]

    prep = MountainCarPreprocessor()
    sast[:,:3] = prep.preprocess(sast[:,:3])

    # select reward
    r = data[:, rewardpos]

    nExperiments = 1
    nIterations = 20
    discRewardsPerExperiment = np.zeros((nExperiments, nIterations))
    for exp in xrange(nExperiments):
        print('Experiment: ' + str(exp))
        estimator = 'wide'
        if estimator == 'extra':
            alg = ExtraTreesRegressor(n_estimators=50, criterion='mse',
                                             min_samples_split=4, min_samples_leaf=2)
            fit_params = dict()
        elif estimator == 'mlp':
            alg = MLP(n_input=sdim+adim, n_output=1, hidden_neurons=5, h_layer=1,
                      optimizer='rmsprop', act_function="sigmoid").getModel()
            fit_params = {'nb_epoch':20, 'batch_size':50, 'validation_split':0.1, 'verbose':1}
            # it is equivalente to call
            #fqi.fit(sast,r,nb_epoch=12,batch_size=50, verbose=1)
        elif estimator == 'incr':
            alg = MergedRegressor(n_input=sdim+adim, n_output=1,
                                hidden_neurons=[5] * (nIterations + 1),
                                n_h_layer_beginning=2,
                                optimizer='rmsprop',
                                act_function=['sigmoid'] * (nIterations + 1),
                                reLearn=False)
            fit_params = {'nb_epoch':20, 'batch_size':50, 'validation_split':0.1, 'verbose':1}
        elif estimator == 'wide':
            alg = WideRegressor(n_input=sdim+adim, n_output=1,
                                hidden_neurons=[5] * (nIterations + 1),
                                n_h_layer_beginning=1,
                                optimizer='rmsprop',
                                act_function=['sigmoid'] * (nIterations + 1),
                                reLearn=False)
            fit_params = {'nb_epoch':20, 'batch_size':50, 'validation_split':0.1, 'verbose':1}
        else:
            raise ValueError('Unknown estimator type.')

        actions = 2
        fqi = FQI(estimator=alg,
                  stateDim=sdim, actionDim=adim,
                  discrete_actions=actions,
                  gamma=0.95, horizon=10, verbose=1,
                  scaled=True)
        #fqi.fit(sast, r, **fit_params)
    
        environment = MountainCar()
        
        discRewards = np.zeros((289, nIterations))
        fqi.partial_fit(sast, r, **fit_params)
        counter = 0
        for i in xrange(-8, 9):
            for j in xrange(-8, 9):
                position = 0.125 * i
                velocity = 0.375 * j
                ## test on the simulator
                environment.reset(position, velocity)
                tupla, rhistory = runEpisode(fqi, environment, environment.gamma)
                #plt.scatter(np.arange(len(rhistory)), np.array(rhistory))
                #plt.show()
                discRewards[counter, 0] = tupla[1]
                counter += 1
        for t in xrange(1, nIterations):
            fqi.partial_fit(None, None, **fit_params)
            mod = fqi.estimator
            counter = 0
            for i in xrange(-8, 9):
                for j in xrange(-8, 9):
                    position = 0.125 * i
                    velocity = 0.375 * j
                    ## test on the simulator
                    environment.reset(position, velocity)
                    tupla, rhistory = runEpisode(fqi, environment, environment.gamma)
                    #plt.scatter(np.arange(len(rhistory)), np.array(rhistory))
                    #plt.show()
                
                    discRewards[counter, t] = tupla[1]
                    counter += 1
        discRewardsPerExperiment[exp, :] = np.mean(discRewards, axis=0)
                
    np.save(estimator, discRewardsPerExperiment)