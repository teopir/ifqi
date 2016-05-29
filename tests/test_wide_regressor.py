from __future__ import print_function

import os
import sys
from pickletools import optimize
sys.path.append(os.path.abspath('../'))

import unittest
import numpy as np
from ifqi.models.incr import WideRegressor
from keras.optimizers import RMSprop


class TestWideRegressor(unittest.TestCase):
    
    def test_1(self):
        reg = WideRegressor(n_input=2,
                           n_output=1,
                           hidden_neurons=[3, 2, 5, 4, 3],
                           n_h_layer_beginning=1,
                           act_function=["sigmoid", "sigmoid", "sigmoid", "sigmoid", "sigmoid"],
                           reLearn=False,
                           optimizer=RMSprop()
                           )
        
        model = reg.model
        weights = model.get_weights()

        self.assertTrue(len(weights) == 4)
        self.assertTrue(weights[0].shape == (2, 3))
        self.assertTrue(weights[1].shape == (3,))
        self.assertTrue(weights[2].shape == (3, 1))
        self.assertTrue(weights[3].shape == (1,))
        
        A1 = np.array([[0, 1, 2], [3, 4, 5]])
        b1 = np.array([0, 7, 54])
        A2 = np.array([[3], [5], [9]])
        b2 = np.array([0.4])
        A3 = np.array([[1], [1]])
        b3 = np.array([0])
        
        params = [A1.copy(), b1.copy(), A2.copy(), b2.copy()]
        
        reg.model.set_weights(params)
        
        reg.adapt(1)

        model = reg.model
        weights = model.get_weights()
        self.assertTrue(len(weights) == 10, '10 != {}'.format(len(weights)))
        # \
        self.assertTrue(weights[0].shape == (2, 3))
        self.assertTrue(np.allclose(weights[0], A1))
        self.assertTrue(weights[1].shape == (3,))
        self.assertTrue(np.allclose(weights[1], b1))

        self.assertTrue(weights[2].shape == (2, 2))
        self.assertTrue(weights[3].shape == (2,))
        
        self.assertTrue(weights[4].shape == (3, 1))
        self.assertTrue(np.allclose(weights[4], A2))
        self.assertTrue(weights[5].shape == (1,))
        self.assertTrue(np.allclose(weights[5], b2))

        self.assertTrue(weights[6].shape == (2, 1))
        self.assertTrue(weights[7].shape == (1,))
        
        self.assertTrue(weights[8].shape == (2, 1))
        self.assertTrue(np.allclose(weights[8], A3))
        self.assertTrue(weights[9].shape == (1,))
        self.assertTrue(np.allclose(weights[9], b3))
        
        print('-'*10, 'Increment 2')
        
        A11 = np.array([[0, 1, 2], [3, 4, 5]])
        b11 = np.array([0, 7, 54])
        A12 = np.array([[3, 1], [1, 2]])
        b12 = np.array([1, 2])
        A21 = np.array([[3], [5], [9]])
        b21 = np.array([0.4])
        A22 = np.array([[1], [2]])
        b22 = np.array([0.3])
        A3 = np.array([[1], [1]])
        b3 = np.array([0])
        A4 = np.array([[1], [1], [1]])
        b4 = np.array([0])

        params = [A11.copy(),
                  b11.copy(),
                  A12.copy(),
                  b12.copy(),
                  A21.copy(),
                  b21.copy(),
                  A22.copy(),
                  b22.copy(),
                  A3.copy(),
                  b3.copy()]
        
        reg.model.set_weights(params)
        
        reg.adapt(2)

        model = reg.model
        weights = model.get_weights()
        self.assertTrue(len(weights) == 14, '14 != {}'.format(len(weights)))
        
        self.assertTrue(weights[0].shape == (2, 3))
        self.assertTrue(np.allclose(weights[0], A11))
        self.assertTrue(weights[1].shape == (3,))
        self.assertTrue(np.allclose(weights[1], b11))
        self.assertTrue(weights[2].shape == (2, 2))
        self.assertTrue(np.allclose(weights[2], A12))
        self.assertTrue(weights[3].shape == (2,))
        self.assertTrue(np.allclose(weights[3], b12))

        self.assertTrue(weights[4].shape == (3, 1))
        self.assertTrue(np.allclose(weights[4], A21))
        self.assertTrue(weights[5].shape == (1,))
        self.assertTrue(np.allclose(weights[5], b21))
        self.assertTrue(weights[6].shape == (2, 1))
        self.assertTrue(np.allclose(weights[6], A22))
        self.assertTrue(weights[7].shape == (1,))
        self.assertTrue(np.allclose(weights[7], b22))
        
        self.assertTrue(weights[8].shape == (2, 5))
        self.assertTrue(weights[9].shape == (5,))
        
        self.assertTrue(weights[10].shape == (5, 1))
        self.assertTrue(weights[11].shape == (1,))
        
        self.assertTrue(weights[12].shape == (3, 1))
        self.assertTrue(np.allclose(weights[12], A4))
        self.assertTrue(weights[13].shape == (1,))
        self.assertTrue(np.allclose(weights[13], b4))

        reg.adapt(3)

if __name__ == '__main__':
    unittest.main()