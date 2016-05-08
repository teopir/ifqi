from __future__ import print_function

# import os
# import sys
# sys.path.append(os.path.abspath('../'))

import unittest
import json
import numpy as np
from ifqi.collectors.collectors import GymCollector

class TestCollector(unittest.TestCase):

    def testpendulum(self):
        gat = GymCollector('Pendulum-v0')
        print(gat)
        nbep = 500
        ms = 300
        dataList, ns, na, nr = gat.collect(max_steps=ms, nbepisodes=nbep, render=False)

        for el in dataList:
            self.assertTrue(len(el) == ms)
            for s in el:
                self.assertTrue(len(s), ns*2+na+nr+1)

        self.assertTrue(len(dataList) == nbep)
        self.assertTrue(ns == 3)
        self.assertTrue(na == 1)
        self.assertTrue(nr == 1)

        d = {'statedim': ns,
             'actiondim': na,
             'rewarddim': nr,
             'data': dataList}

        with open('data.txt', 'w') as outfile:
            json.dump(d, outfile)

if __name__ == '__main__':
    unittest.main()