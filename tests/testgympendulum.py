from __future__ import print_function

import os
import sys
sys.path.append(os.path.abspath('../'))

import unittest
import numpy as np
from ifqi.fqi.FQI import FQI
from ifqi.models.mlp import MLP
import ifqi.utils.parser as parser
from sklearn.ensemble import ExtraTreesRegressor

import pickle

class TestGymPendulum(unittest.TestCase):

    # def testextratrees(self):
    #     data, sdim, adim, rdim = parser.parsejson('pendulum500steps1000ep.json')
    #     self.assertTrue(sdim == 3)
    #     self.assertTrue(adim == 1)
    #     self.assertTrue(rdim == 1)
    #
    #     alg = ExtraTreesRegressor(n_estimators=50, criterion='mse',
    #                                      min_samples_split=2, min_samples_leaf=1)
    #
    #     fqi = FQI(alg, sdim, adim, verbose=1)
    #     rewardpos = sdim + adim
    #     indicies = np.delete(np.arange(data.shape[1]), rewardpos)
    #     print(indicies)
    #     sast = data[:, indicies]
    #     r = data[:, rewardpos]
    #     print(sast.shape)
    #     fqi.fit(sast, r)

    def testmlp(self):
        data, sdim, adim, rdim = parser.parsejson('pendulum500steps1000ep.json')
        self.assertTrue(sdim == 3)
        self.assertTrue(adim == 1)
        self.assertTrue(rdim == 1)

        alg = MLP(n_input=sdim+adim, optimizer='rmsprop').getModel()

        fqi = FQI(alg, sdim, adim, verbose=1)
        rewardpos = sdim + adim
        indicies = np.delete(np.arange(data.shape[1]), rewardpos)
        print(indicies)
        sast = data[:, indicies]
        r = data[:, rewardpos]
        print(sast.shape)
        fqi.fit(sast, r, nb_epoch=10, batch_size=50, verbose=1)


if __name__ == '__main__':
    unittest.main()
