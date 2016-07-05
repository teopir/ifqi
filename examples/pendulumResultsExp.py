"""PendulumResultsExp.py

This source code (interfaced with ExpMan) run FQI with a given dataset and a given model.
It provides the results to ExpMan library, which collects all the results from different running and automatically
plot some statistics
"""


from __future__ import print_function
from ifqi.models.loss_history import LossHistory
import os
import sys
#sys.path.append(os.path.abspath('../'))

import numpy as np
from ifqi.fqi.FQI import FQI
from ifqi.models.mlp import MLP
from ifqi.models.incr import IncRegression, MergedRegressor, WideRegressor
from ifqi.envs.invertedPendulum import InvPendulum
from ifqi.envs.bicycle import Bicycle
import utils.net_serializer as serializer
import utils.QSerTest as qserializer
import ifqi.utils.parser as parser
from sklearn.ensemble import ExtraTreesRegressor
import time
from keras.regularizers import l2
import datetime
import matplotlib.pyplot as plt
from ifqi.envs.mountainCar import MountainCar

#https://github.com/SamuelePolimi/ExperimentManager.git
from ExpMan.core.ExperimentManager import ExperimentManager

"""---------------------------------------------------------------------------
Retrive parameter and load the ExpMan.core.ExperimentSample
---------------------------------------------------------------------------"""

def retreiveParams():
    global folder,test,case,code, number
    if len(sys.argv) < 6:
        raise "Number of parameters not sufficient"
    folder = sys.argv[1]
    test = int(sys.argv[2])
    case = sys.argv[3]
    code = sys.argv[4]
    number = int(sys.argv[5])
    
    for arg in sys.argv[6:]:
        s = arg.split("=")    
        name = s[0]
        value = s[1]
        try:
            if "." in value:
                globals()[name] = float(value)
            else:
                globals()[name] = int(value)
        except:
            globals()[name] = str(value)
            
def loadSample():
    global folder, test, case, code, number
    man = ExperimentManager()
    man.loadExperiment(folder, verbose=False)
    expTest = man.openExperimentTest(test)
    expTest.loadTestCases()
    expCase = expTest.openTestCase(case)
    expCase.loadSamples()
    sample = expCase.openSample(code, number)
    return sample
    
"""---------------------------------------------------------------------------
Let's run the episode
---------------------------------------------------------------------------"""
    
def runCarEpisode(myfqi, environment, gamma):   # (nstep, J, success)
    J = 0
    t = 0
    test_succesful = 0
    rh = []
    while(t < 500 and not environment.isAbsorbing()):
        state = environment.getState()
        action, _ = myfqi.predict(np.array(state))
        position, velocity, r = environment.step(action)
        J += gamma ** t * r
        t += 1
        rh += [r]
        
        if r == 1:
            print('Goal reached')
            test_succesful = 1

    return (t, J, test_succesful), rh

def runBicycleEpisode(myfqi, environment, gamma):   # (nstep, J, success)
    J = 0
    t = 0
    test_succesful = 0
    rh = []
    while(not environment.isAbsorbing()):
        state = environment.getState()
        action, _ = myfqi.predict(np.array(state))
        r = environment.step(action[0])
        J += gamma ** t * r
        t += 1
        rh += [r]
        
        if environment.isAtGoal():
            print('Goal reached')
            test_succesful = 1

    return (t, J, test_succesful), rh
    
def runEpisode(myfqi, environment, gamma):   # (nstep, J, success)
    J = 0
    t=0
    test_succesful = 0
    rh = []
    while(not environment.isAbsorbing()):
        state = environment.getState()
        action, _ = myfqi.predict(np.array(state))
        r = environment.step(action)
        J += gamma**t * r
        if(t>3000):
            print('COOL! you done it!')
            test_succesful = 1
            break
        t+=1
        rh += [r]
    return (t, J, test_succesful), rh

       
"""---------------------------------------------------------------------------
Init all the parameters
---------------------------------------------------------------------------"""

#which dataset
dataset = "A"
nsample=0


#network general configuration
#This will be overwritten by retreiveParams()
n_neuron = 5
n_layer = 2
n_increment = 2
activation = "sigmoid"
inc_activation = "sigmoid"
n_epoch = 300
batch_size = 100
n_iter = 5
model = "mlp"
env = "pen"#"car"#"bic"
scaled=False

#plotting configuration
last_layer = False
qfunction = False
policy = False

#variable to save
step = []                                               #history dim=0
J = []                                                  #history dim=0
goal = 0                                                #single dim=0
loss = [[]]                                             #history dim=1 (we will plot a video)
last_loss = []

"""---------------------------------------------------------------------------
Connection with the test
---------------------------------------------------------------------------"""

retreiveParams()
sample = loadSample()

"""---------------------------------------------------------------------------
Init the regressor
---------------------------------------------------------------------------"""
#-----------------------------------------------------------------------------
#Retrive the data
#-----------------------------------------------------------------------------

if(env=="pen"):
    test_file = "dataset/pendulum_data/"+dataset+"/data"+str(nsample)+".log"
elif(env=="car"):
    test_file = 'dataset/mc/mc_0.log'
else:
    test_file = 'dataset/bicycle_data/bicycle' + str(nsample) +'.txt'
    
data, sdim, adim, rdim = parser.parseReLeDataset(test_file)

estimator = model
history = LossHistory()

if estimator == 'extra':
    alg = ExtraTreesRegressor(n_estimators=50, criterion='mse',
                                     min_samples_split=2, min_samples_leaf=1)
    fit_params = {"callbacks":[history]}
elif estimator == 'mlp':
    alg = MLP(n_input=sdim+adim, n_output=1, hidden_neurons=n_neuron, h_layer=n_layer,
              optimizer='rmsprop', act_function=activation)
    fit_params = {'nb_epoch':n_epoch, 'batch_size':batch_size, 'verbose':0,"callbacks":[history]}
elif estimator == 'frozen-incr':
    alg = MergedRegressor(n_input=sdim+adim, n_output=1, hidden_neurons=[n_neuron]*(n_iter+2), 
              n_h_layer_beginning=n_layer,optimizer='rmsprop', act_function=[activation]*2+[inc_activation]*(n_iter*n_increment))
    fit_params = {'nb_epoch':n_epoch, 'batch_size':batch_size, 'verbose':0,"callbacks":[history]}
elif estimator == 'unfrozen-incr':
    alg = MergedRegressor(n_input=sdim+adim, n_output=1, hidden_neurons=[n_neuron]*(n_iter+2), 
              n_h_layer_beginning=n_layer,optimizer='rmsprop', act_function=[activation]*2+[inc_activation]*(n_iter*n_increment),reLearn=True)
    fit_params = {'nb_epoch':n_epoch, 'batch_size':batch_size, 'verbose':0,"callbacks":[history]}
elif estimator == 'wide':
    alg = WideRegressor(n_input=sdim+adim, n_output=1, hidden_neurons=[n_neuron]*(n_iter+2), 
              n_h_layer_beginning=n_layer,optimizer='rmsprop', act_function=[activation]*2+[inc_activation]*(n_iter*n_increment))
    fit_params = {'nb_epoch':n_epoch, 'batch_size':batch_size, 'verbose':0,"callbacks":[history]}
else:
    raise ValueError('Unknown estimator type.')

"""---------------------------------------------------------------------------
The experiment phase
----------------------------------------------------------------------------"""


len_data = data.size


rewardpos = sdim + adim
indicies = np.delete(np.arange(data.shape[1]), rewardpos)

#-----------------------------------------------------------------------------
#Prepare sast matrix
#-----------------------------------------------------------------------------

# select state, action, nextstate, absorbin
sast = data[:, indicies]

# select reward
r = data[:, rewardpos]

if(env=="pen"):            
    actions = (np.arange(3)).tolist()
elif(env=="car"):
    actions = (np.arange(2)).tolist()
else:
    actions = (np.arange(9)).tolist()


#initialize folder
if not os.path.exists(sample.path):
    os.makedirs(sample.path)
            
#-----------------------------------------------------------------------------
#Run FQI
#-----------------------------------------------------------------------------

fqi = FQI(estimator=alg,
          stateDim=sdim, actionDim=adim,
          discrete_actions=actions,
          gamma=0.95,scaled=scaled, horizon=31, verbose=1)
          
if(env=="pen"):
    environment = InvPendulum()
elif(env=="car"):
    environment = MountainCar()
else:
    environment = Bicycle()

for iteration in range(n_iter):
                
    #fit
    if(iteration==0):
        fqi.partial_fit(sast, r, **fit_params)
    else:
        fqi.partial_fit(None, None, **fit_params)
        
    mod = fqi.estimator
    
    ## test on the simulator
    environment.reset()
    print("Environment Execution")
    if(env=="pen"):
        tupla, rhistory = runEpisode(fqi, environment, 0.95)
    elif(env=="car"):
        tupla, rhistory = runCarEpisode(fqi, environment, 0.95)
    else:
        tupla, rhistory = runBicycleEpisode(fqi,environment,0.95)
    
    print("End Environment Execution")
    t, j, s = tupla
    
    #keep track of the results
    step.append(t) 
    J.append(j)
    last_loss.append(history.losses[-1].tolist())
    
    #loss.append(rhistory)
    if(s==1):
        goal=1
        
    

"""---------------------------------------------------------------------------
Saving phase
---------------------------------------------------------------------------"""

sample.addVariableResults("step", step)
sample.addVariableResults("J", J)
sample.addVariableResults("last_loss", last_loss)
#sample.addVariableResult("loss",loss)
sample.addVariableResults("goal",goal)

sample.closeSample()
