from __future__ import print_function

import os
import sys
sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('fqi'))

#test imports
from fqi_trees import FQI as FQITree

# my imports
import unittest
import numpy as np
from ifqi.fqi.FQI import FQI
import ifqi.utils.parser as parser
from ifqi.preprocessors.invPendulumPreprocessor import InvertedPendulumPreprocessor
from sklearn.ensemble import ExtraTreesRegressor
from ifqi.envs.actor import Actor
from ifqi.envs.invertedPendulum import InvPendulum


def runEpisode(obj, actor, myfqi, environment, gamma):   # (nstep, J, success)
    J = 0
    t=0
    test_succesful = 0
    while(not environment.isAbsorbing()):
        state = environment.getState()
        action = actor.exploitAction(state)
        myaction, _ = myfqi.predict(np.array(state))
        obj.assertTrue(np.allclose(action, myaction+1))
        J += gamma**t * environment.step(action)
        if(t>500):
            test_succesful = 1
            break
        t+=1
    return (t, J, test_succesful)


class TestFQIs(unittest.TestCase):

    def testcomparefqiextra(self):
        #filename = '../../dataset/pendulum/A/data0.log'
        filename = '../../dataset/episodicPendulum.txt'
        nActions = 3
        ###########
        # define other fqi
        fqi = FQITree(nActions, gamma=.95, nIterations=20, nb_epoch=10000, batch_size=100)

        ###########
        # define my fqi
        alg = ExtraTreesRegressor(n_estimators=50, criterion='mse',
                                         min_samples_split=2, min_samples_leaf=1,
                                  random_state = 54)

        actions = (np.arange(nActions+1) - 1).tolist()
        myfqi = FQI(estimator=alg,
                    stateDim=2, actionDim=1,
                    discrete_actions=actions,
                    gamma=0.95, horizon=6, verbose=1, scaled=False)

        gPrepro = InvertedPendulumPreprocessor()

        ###########
        # read dataset
        sars = fqi.parseDataset(filename)
        data, sdim, adim, rdim = parser.parseReLeDataset(filename)
        self.assertTrue(np.allclose(sars,data))

        ############
        # prepare data for my fqi
        rewardpos = sdim + adim
        indicies = np.delete(np.arange(data.shape[1]), rewardpos)
        sast = data[:, indicies]
        sast[:, :3] = gPrepro.preprocess(sast[:, :3])
        # select reward
        r = data[:, rewardpos]


        X, y = fqi.first_run(sars, preprocessor=gPrepro, save=False)
        myX, myy = myfqi._partial_fit(sast, r)
        self.assertTrue(np.allclose(X, myX),
                        '{} != {}'.format(X, myX))
        self.assertTrue(np.allclose(y, myy),
                        '{} != {}'.format(y, myy))

        mod = fqi.model
        mymod = myfqi.estimator

        def check_extra_model(obj, moda, modb):
            obj.assertTrue(np.allclose(moda.n_features_, modb.n_features_),
                           '{} != {}'.format(moda.n_features_, modb.n_features_))
            obj.assertTrue(np.allclose(moda.n_outputs_, modb.n_outputs_),
                           '{} != {}'.format(moda.n_outputs_, modb.n_outputs_))
            obj.assertTrue(np.allclose(moda.feature_importances_, modb.feature_importances_),
                           '{} != {}'.format(moda.feature_importances_, modb.feature_importances_))

            for a in myfqi._actions:
                act = np.ones((sars.shape[0], 1)) * a[0]
                tmp = np.concatenate((sars[:, :2], act),axis=1)
                pred = moda.predict(tmp)
                mypred = modb.predict(tmp)
                #print(np.concatenate((pred.reshape(-1,1),mypred.reshape(-1,1)), axis=1))
                obj.assertTrue(np.allclose(pred,mypred))
                obj.assertTrue(np.allclose(moda.score(tmp,pred), modb.score(tmp,mypred)))

        check_extra_model(self, mod, mymod)

        environment = InvPendulum()
        for iteration in range(myfqi.horizon):
            X, y = fqi.run()
            myX, myy = myfqi._partial_fit(sast, r)
            self.assertTrue(np.allclose(y, myy),
                            '{} != {}'.format(y, myy))
            self.assertTrue(np.allclose(X, myX),
                            '{} != {}'.format(X, myX))
            check_extra_model(self, fqi.model, myfqi.estimator)

            print('Performing simulation')
            actor = Actor(fqi.model, epsilon=0, n_state=2, n_act=3)
            environment.reset()
            tupla = runEpisode(self, actor, myfqi, environment, 0.95)

if __name__ == '__main__':
    unittest.main()