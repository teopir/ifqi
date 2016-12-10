from __future__ import print_function
from ifqi.algorithms.pbo.gradpbo import GradPBO
import numpy as np

gpbo = GradPBO()

rho = np.array([1,2]).reshape(1,2)
theta = np.array([1, 2]).reshape(-1,1)
print(rho.shape)
print(theta.shape)
v = gpbo.bopf(rho, theta)
print(v)

s = np.array([1,2,3]).reshape(-1,1)
a = np.array([0,3,4]).reshape(-1,1)
q = gpbo.qf(s,a, theta)
print(q)