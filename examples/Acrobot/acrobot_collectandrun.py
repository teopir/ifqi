from __future__ import print_function
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath('../'))
from context import *

from ifqi.envs.acrobot import Acrobot

env = Acrobot()

class tmp_policy():
    def drawAction(self, state):
        return -5 + 10 * np.random.rand()

##############################################################
# Compute the discounted reward
n_rep = 1
pol = tmp_policy()
Jsample = env.evaluate(pol, nbEpisodes=n_rep, metric='discounted', render=True)
print(Jsample)

##############################################################
# Collect samples
A = env.collectEpisode(None)
print(A.shape)
