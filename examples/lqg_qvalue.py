from __future__ import print_function
from context import *
import numpy as np

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from ifqi.envs.lqg1d import LQG1D

mdp = LQG1D()
K = mdp.computeOptimalK()
print('Optimal K: {}'.format(K))
xs = np.linspace(-mdp.max_pos, mdp.max_pos, 60)
us = np.linspace(-mdp.max_action, mdp.max_action, 50)

l = []
for x in xs:
    for u in us:
        v = mdp.computeQFunction(x, u, K, 0.0001, 100)
        l.append([x, u, v])
tabular_Q = np.array(l)
np.savetxt('lqg_qtable.txt', tabular_Q)
print('printed Q-table')

# print(tabular_Q.shape)
# print(tabular_Q)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(tabular_Q[:,0], tabular_Q[:,1], tabular_Q[:,2])
plt.show()


