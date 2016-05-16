from __future__ import print_function

import os
import sys
from pickletools import optimize
sys.path.append(os.path.abspath('../'))

import unittest
import numpy as np
from ifqi.models.incr import MergedRegressor
from keras.optimizers import RMSprop


class TestMergeRegressor(unittest.TestCase):
    
    def test_1(self):
        reg = MergedRegressor(n_input=2, n_output=1,
                              hidden_neurons=[3,2, 5],
                              n_h_layer_beginning=1,
                              act_function=["sigmoid","sigmoid","relu"],
                              reLearn=False, optimizer=RMSprop()
                              )
        
        model = reg.model
        weights = model.get_weights()
        self.assertTrue(len(weights) == 4)
        self.assertTrue(weights[0].shape == (2,3))
        self.assertTrue(weights[1].shape == (3,))
        self.assertTrue(weights[2].shape == (3,1))
        self.assertTrue(weights[3].shape == (1,))
        
        A1 = np.array([[0,1,2],[3,4,5]])
        b1 = np.array([0,7,54])
        A2 = np.array([[3],[5],[9]])
        b2 = np.array([0.4])
        
        params = [
                  A1.copy(), b1.copy(), A2.copy(), b2.copy()        
                  ]
        
        reg.model.set_weights(params)
        
        reg.adapt(1)
        model = reg.model
        weights = model.get_weights()
        self.assertTrue(len(weights) == 8, '8 != {}'.format(len(weights)))
        # \
        self.assertTrue(weights[0].shape == (2,3))
        self.assertTrue(np.allclose(weights[0], A1))
        self.assertTrue(weights[1].shape == (3,))
        self.assertTrue(np.allclose(weights[1], b1))
        self.assertTrue(weights[2].shape == (3,1))
        self.assertTrue(np.allclose(weights[2], A2))
        self.assertTrue(weights[3].shape == (1,))
        self.assertTrue(np.allclose(weights[3], b2))
        # 
        self.assertTrue(weights[4].shape == (3,2), '{}'.format(weights[4].shape))
        self.assertTrue(weights[5].shape == (2,))
        self.assertTrue(weights[6].shape == (3,1), '{}'.format(weights[6].shape))
        self.assertTrue(weights[7].shape == (1,))
        
        print('-'*10, 'Increment 2')
        reg.adapt(2)
        model = reg.model
        weights = model.get_weights()
        self.assertTrue(len(weights) == 12, '12 != {}'.format(len(weights)))
        # left model
        self.assertTrue(weights[0].shape == (2,3))
        self.assertTrue(np.allclose(weights[0], A1))
        self.assertTrue(weights[1].shape == (3,))
        self.assertTrue(np.allclose(weights[1], b1))
        self.assertTrue(weights[2].shape == (3,1))
        self.assertTrue(np.allclose(weights[2], A2))
        self.assertTrue(weights[3].shape == (1,))
        self.assertTrue(np.allclose(weights[3], b2))
        #
        self.assertTrue(weights[4].shape == (3,2), '{}'.format(weights[4].shape))
        self.assertTrue(weights[5].shape == (2,))
        self.assertTrue(weights[6].shape == (3,1), '{}'.format(weights[6].shape))
        self.assertTrue(weights[7].shape == (1,))
        #
        self.assertTrue(weights[8].shape == (3,5))
        self.assertTrue(weights[9].shape == (5,))
        self.assertTrue(weights[10].shape == (6,1))
        self.assertTrue(weights[11].shape == (1,))
        

if __name__ == '__main__':
    unittest.main()