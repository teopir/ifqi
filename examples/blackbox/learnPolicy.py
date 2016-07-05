from __future__ import print_function

import os
import sys
#sys.path.append(os.path.abspath('../'))

import numpy as np
from ifqi.fqi.FQI import FQI #I think you can safely use fqi.FQI
from ifqi.models.mlp import MLP
from ifqi.models.incr import IncRegression, MergedRegressor
from ifqi.models.incr_inverse import ReverseIncRegression
from ifqi.envs.blackbox import BlackBox
import utils.net_serializer as serializer
import utils.QSerTest as qserializer
import ifqi.utils.parser as parser
from sklearn.ensemble import ExtraTreesRegressor
import time
from keras.regularizers import l2
import datetime
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
import gc
from sklearn.externals import joblib

#https://github.com/SamuelePolimi/ExperimentManager.git
from ExpMan.core.ExperimentManager import ExperimentManager

def runEpisode(myfqi, environment, gamma):   # (nstep, J, success)

    J = 0
    t=0
    test_succesful = 0
    rh = []
    time = 0
    while environment.next:
        state = environment.getState()
        action, _ = myfqi.predict(np.array(state))
        #action=0
        
        while environment.next:
            time+=1
            if time%100==0:
                break
            r = environment.step(action)
            #print ("score" ,t)
            J += gamma**t * r
            t+=r
        #rh.append(r)
    #print("time: " + str(time))
    return (t, J, test_succesful), rh
    
    
iter_ = raw_input("Which iter would you like to do?")

sast = np.load("results/BlackBox/test3/extra/linearmodel/0" + "/fqidata" + str(0) + ".npy")
#print ("sast.shape",sast.shape)
r = np.load("results/BlackBox/test3/extra/linearmodel/0" + "/rewards" + str(0)+ ".npy")
for i in range(1, int(iter_)+1):
    
    sast = np.concatenate((sast,np.load("results/BlackBox/test3/extra/linearmodel/0" + "/fqidata" + str(iter_) + ".npy")),axis=0)
    #print ("sast.shape",sast.shape)
    r = np.concatenate((r,np.load("results/BlackBox/test3/extra/linearmodel/0" + "/rewards" + str(iter_)+ ".npy")),axis=0)

    

print ("r.shape",np.matrix(r).T.shape)
sast_r = np.concatenate((sast,np.matrix(r).T),axis=1)
print("sast_r.shape", sast_r.shape)
np.random.shuffle(sast_r)

data_size = min(sast_r.shape[0],100000)
#subsamp
all_states = sast_r[:data_size,:36]

#(50000,1)
#(1,50000)
sast = sast_r[:data_size,:-1]
print ("sast.shape",sast.shape)
r = sast_r[:data_size,-1]
print ("r.shape",np.asarray(r).ravel().shape)
r = np.asarray(r).ravel() 
#r = np.asarray(r).ravel() #+ np.random.randn(data_size,) * np.std(r) * 0.1

alg = ExtraTreesRegressor(n_estimators=50, criterion='mse',
                                     min_samples_split=8, min_samples_leaf=4)
                                     
fit_params = dict()
actions = (np.arange(4)).tolist()
fqi = FQI(estimator=alg,
          stateDim=36, actionDim=1,
          discrete_actions=actions,
          #TODO: set scaled again
          gamma=0.99,scaled=False, horizon=31, verbose=1)    

environment = BlackBox(data_file="train")
    
n_iter = 10
l1=[0]*(n_iter-1)
l2=[0]*(n_iter-1)
linf=[0]*(n_iter-1)
    
sast_dim = sast.shape[0]
new_q = np.zeros((4,sast_dim))
#TODO:remove
t=0
t_best=0
for iteration in range(n_iter):
    
    
    #-----------------------------------------------------------------------------
    #Run FQI
    #-----------------------------------------------------------------------------
                                    
    #fit
    if(iteration==0):
        fqi.partial_fit(sast, r, **fit_params)
    else:
        fqi.partial_fit(None, None, **fit_params)

    mod = fqi.estimator
    

    print("Running experiment")
    (t, J , _), _ = runEpisode(fqi, environment, 0.99)  # (nstep, J, success)
    t = environment.get_score()
    if t > t_best:
        print("Best results ever! SAVE")
        joblib.dump(alg, 'extratree.pkl') 
        t_best=t
    environment.reset()
    print("total score: " + str(t))
        