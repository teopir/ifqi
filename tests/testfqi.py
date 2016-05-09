from __future__ import print_function

# import os
# import sys
# sys.path.append(os.path.abspath('../'))

import unittest
import numpy as np
from ifqi.fqi.FQI import FQI
from ifqi.models.mlp import MLP
import ifqi.utils.parser as parser
from sklearn.ensemble import ExtraTreesRegressor

import pickle

class TestFQI(unittest.TestCase):

    def testMaxQA(self):
        nstates_dim = 2
        nactions_dim = 1
        nactions = 3
        nbsamples = 3*nactions

        st0 = np.random.get_state()
        # with open('company_data.pkl', 'wb') as output:
        #     pickle.dump(st0, output)

        # with open('company_data.pkl', 'rb') as input:
        #     st0 = pickle.load(input)
        #     np.random.set_state(st0)

        class dummyestimator:
            def predict(self, X):
                action = X[0,-1]
                v = np.ones((X.shape[0],1)) * -9
                idx = int(nbsamples/nactions)
                off = 0
                for a in range(nactions):
                    print(action,a)
                    if action == a:
                        v[off:off+idx] = action
                    off = off+idx
                return v

        X = np.random.randint(-1, 10, (nbsamples, nstates_dim+nactions_dim+nstates_dim+1))
        # define absorbing state
        X[:, -1] = np.random.randint(0, 2, (nbsamples, ))
        # define actions
        X[:, 2] = np.random.randint(0, nactions, (nbsamples, ))

        print('---- DATA ----')
        print(X)
        print()

        de = dummyestimator()
        fqi = FQI(de, nstates_dim, nactions_dim)
        print(fqi.__name__)

        fqi._actions = np.arange(0, nactions).reshape(-1,1)
        print(fqi._actions)

        Q, A = fqi.maxQA(X[:, 3:5])

        self.assertTrue(Q.size == nbsamples, 'wrong dimension of Q')
        self.assertTrue(A.size == Q.size, 'wrong dimension of A')
        idx = int(nbsamples/nactions)
        off = 0
        for i in range(nactions):
            self.assertTrue(np.allclose(Q[off:off+idx],i), '{}'.format(Q))
            self.assertTrue(np.allclose(A[off:off+idx],i), '{}'.format(A))
            off = off + idx

        Q, A = fqi.maxQA(X[:, 3:5], X[:,-1])

        self.assertTrue(Q.size == nbsamples, 'wrong dimension of Q')
        self.assertTrue(A.size == Q.size, 'wrong dimension of A')
        off = 0
        for i in range(nactions):
            for r in range(off, min(off+idx, nbsamples)):
                if X[r,-1] == 1:
                    self.assertTrue(Q[r] == 0)
                    self.assertTrue(A[r] == 0)
                else:
                    self.assertTrue(Q[r] == i)
                    self.assertTrue(A[r] == i)
            off = off + idx

    def testmlp(self):
        data, sdim, adim, rdim = parser.parseReLeDataset('../dataset/episodicPendulum.txt')
        self.assertTrue(sdim == 2)
        self.assertTrue(adim == 1)
        self.assertTrue(rdim == 1)

        alg = MLP(n_input=sdim+adim, optimizer='rmsprop').getModel()

        fqi = FQI(alg, sdim, adim, verbose=1)
        sast = data[:, [0,1,2,4,5,6]]
        r = data[:, 3]
        print(sast.shape)
        fqi.fit(sast, r, nb_epoch=22, batch_size=32, verbose=0)

    def testextratrees(self):
        data, sdim, adim, rdim = parser.parseReLeDataset('../dataset/episodicPendulum.txt')
        self.assertTrue(sdim == 2)
        self.assertTrue(adim == 1)
        self.assertTrue(rdim == 1)

        alg = ExtraTreesRegressor(n_estimators=50, criterion='mse',
                                         min_samples_split=2, min_samples_leaf=1)

        fqi = FQI(alg, sdim, adim, verbose=1)
        sast = data[:, [0,1,2,4,5,6]]
        r = data[:, 3]
        print(sast.shape)
        fqi.fit(sast, r)

    def testComputeAction(self):
        data, sdim, adim, rdim = parser.parseReLeDataset('../dataset/episodicPendulum.txt')
        self.assertTrue(sdim == 2)
        self.assertTrue(adim == 1)
        self.assertTrue(rdim == 1)

        rewardpos = sdim + adim
        indicies = np.delete(np.arange(data.shape[1]), rewardpos)

        # select state, action, nextstate, absorbin
        sast = data[:, indicies]
        # select reward
        r = data[:, rewardpos]

        alg = ExtraTreesRegressor(n_estimators=50, criterion='mse',
                                         min_samples_split=2, min_samples_leaf=1, verbose=0)
        fit_params = {}
        actions = (np.arange(10) - 1).tolist()
        n_actions = len(actions)
        fqi = FQI(estimator=alg,
              stateDim=sdim, actionDim=adim,
              discrete_actions=actions,
              gamma=0.9, horizon=10, verbose=1)
        fqi._compute_actions(None)
        print(fqi._actions)
        self.assertTrue(isinstance(fqi._actions, np.ndarray))
        self.assertTrue(fqi._actions.shape[0] == n_actions)
        self.assertTrue(fqi._actions.shape[1] == adim)
        for i in range(n_actions):
            self.assertTrue(fqi._actions[i,0] == actions[i], '{} != {}'.format(fqi._actions[i,0], actions[i]))

        fqi = FQI(estimator=alg,
              stateDim=sdim, actionDim=adim,
              discrete_actions=6,
              gamma=0.9, horizon=10, verbose=1)
        #fqi._compute_actions(sast)
        fqi.partial_fit(sast, r)
        print(fqi._actions)
        self.assertTrue(isinstance(fqi._actions, np.ndarray))
        self.assertTrue(fqi._actions.shape[0] == 6)
        self.assertTrue(fqi._actions.shape[1] == adim)
        self.assertTrue(fqi.iteration == 1)

        fqi = FQI(estimator=alg,
              stateDim=sdim, actionDim=adim,
              discrete_actions=[0, 1, 2, 3],
              gamma=0.9, horizon=10, verbose=1)
        fqi.partial_fit(sast, r)
        print(fqi._actions)
        self.assertTrue(isinstance(fqi._actions, np.ndarray))
        self.assertTrue(fqi._actions.shape[0] == 4)
        self.assertTrue(fqi._actions.shape[1] == adim)
        for i in range(4):
            self.assertTrue(fqi._actions[i] == i, '{} != {}'.format(fqi._actions[i], i))
        self.assertTrue(fqi.iteration == 1)
        fqi.partial_fit(sast, r)
        self.assertTrue(fqi.iteration == 2)

        actionpos = 2
        sast[:,actionpos] = np.random.randint(5,8, (sast.shape[0], ))
        fqi = FQI(estimator=alg,
              stateDim=sdim, actionDim=adim,
              discrete_actions=3,
              gamma=0.9, horizon=10, verbose=1)
        fqi.partial_fit(sast, r)
        print(fqi._actions)
        self.assertTrue(isinstance(fqi._actions, np.ndarray))
        self.assertTrue(fqi._actions.shape[0] == 3)
        self.assertTrue(fqi._actions.shape[1] == adim)
        for i in range(3):
            self.assertTrue(fqi._actions[i] == i+5, '{} != {}'.format(fqi._actions[i], i))

if __name__ == '__main__':
    unittest.main()