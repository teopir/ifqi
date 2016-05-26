from __future__ import print_function

import os
import sys
#sys.path.append(os.path.abspath('../'))

import numpy as np
from ifqi.fqi.FQI_sam import FQI #I think you can safely use fqi.FQI
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
            print ("set: " + name + "=" + str(value) )
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

def runEpisode(myfqi, environment, gamma):   # (nstep, J, success)

    J = 0
    t=0
    test_succesful = 0
    rh = []
    time = 0
    while time <1000:
        time+=1
        state = environment.getState()
        action, _ = myfqi.predict(np.array(state))
        #action=0
        r = environment.step(action)
        J += gamma**t * r
        t+=r
        #rh.append(r)
    print("time: " + str(time))
    return (t, J, test_succesful), rh

def runOnTram(myfqi, environment, eps, sast):
    rnd = np.random.rand()
    while environment.next and eps<rnd:
        state = environment.getState()
        action, _ = myfqi.predict(np.array(state))
        environment.step(action)
        next_state =  environment.getState()
        sast.append(state.tolist() + [action] + next_state.tolist() + [1-environment.next])
        rnd = np.random.rand()
    
def runEpisodeToDS(environment, fqi_list,eps_out):   # (nstep, J, success)

    sast = []
    while environment.next:
        state = environment.getState()
        #action, _ = myfqi.predict(np.array(state))
        action = np.random.randint(4 + len(fqi_list))
        if(action >= 4):
            runOnTram(fqi_list[action-4], environment, eps_out, sast)
            continue
        environment.step(action)
        next_state =  environment.getState()
        sast.append(state.tolist() + [action] + next_state.tolist() + [1-environment.next])
    return sast

       
"""---------------------------------------------------------------------------
Init all the parameters
---------------------------------------------------------------------------"""

#which dataset
dataset = "linearmodel"
nsample=0


#network general configuration
#This will be overwritten by retreiveParams()
n_neuron = 50
n_layer = 2
n_increment = 1
activation = "sigmoid"
inc_activation = "sigmoid"
n_epoch = 300
batch_size = 100
n_iter = 15
model = "mlp"
scaled=True
n_estimators=50
data_size = 50000
epsilon = 1         #TODO: set epsilon
eps_out = 0.99
n_cycle = 5

#plotting configuration
last_layer = False
qfunction = False
policy = False

#variable to save
std_scores = []
std_states = []
mean_scores = []

"""---------------------------------------------------------------------------
Connection with the test
---------------------------------------------------------------------------"""

retreiveParams()
sample = loadSample()

"""----------------------------------------------------------------------------
Test changes
----------------------------------------------------------------------------"""
print("effective model: " + model)
"""---------------------------------------------------------------------------
Init the regressor
---------------------------------------------------------------------------"""

sdim=36
adim=1

#print ("model = " + model)
#print ("n_neuron = " + str(n_neuron))


estimator = model
if estimator == 'extra':
    alg = ExtraTreesRegressor(n_estimators=50, criterion='mse',
                                     min_samples_split=2, min_samples_leaf=1)
    fit_params = dict()
elif estimator == 'mlp':
    alg = MLP(n_input=sdim+adim, n_output=1, hidden_neurons=n_neuron, h_layer=n_layer,
              optimizer='rmsprop', act_function=activation).getModel()
    fit_params = {'nb_epoch':n_epoch, 'batch_size':batch_size, 'verbose':0}
elif estimator == 'frozen-incr':
    alg = MergedRegressor(n_input=sdim+adim, n_output=1, hidden_neurons=[n_neuron]*(niter+2), 
              n_h_layer_beginning=n_layer,optimizer='rmsprop', act_function=[activation]*2+[inc_activation]*(niter*n_increment))
    fit_params = {'nb_epoch':n_epoch, 'batch_size':batch_size, 'verbose':0}
elif estimator == 'unfrozen-incr':
    alg = MergedRegressor(n_input=sdim+adim, n_output=1, hidden_neurons=[n_neuron]*(niter+2), 
              n_h_layer_beginning=n_layer,optimizer='rmsprop', act_function=[activation]*2+[inc_activation]*(niter*n_increment),reLearn=True)
    fit_params = {'nb_epoch':n_epoch, 'batch_size':batch_size, 'verbose':0}
elif estimator == 'boost':
    #TODO: implement
    #raise "<boost> Not implemented exception"    
    alg = MLP(n_input=sdim+adim, n_output=1, hidden_neurons=n_neuron, h_layer=n_layer,
              optimizer='rmsprop', act_function=activation).getModel()
    fit_params = {'nb_epoch':n_epoch, 'batch_size':batch_size, 'verbose':0}

else:
    raise ValueError('Unknown estimator type.')

"""---------------------------------------------------------------------------
The experiment phase
----------------------------------------------------------------------------"""


#-----------------------------------------------------------------------------
#Retrive the data
#-----------------------------------------------------------------------------

#test_file = "dataset/pendulum_data/"+dataset+"/data"+str(nsample)+".log"
#data, sdim, adim, rdim = parser.parseReLeDataset(test_file)

data = np.load("dataset/blackbox_data/" + dataset + "/data" + str(nsample) + ".npy")
#subsampling


#-----------------------------------------------------------------------------
#Prepare sast matrix
#-----------------------------------------------------------------------------

np.random.shuffle(data)
data = data[0:data_size,:]
#np.concatenate((data,np.array([0]*data.size).T),axis=1)


len_data = data.size

rewardpos = sdim + adim
stateDim = sdim
indicies = np.delete(np.arange(data.shape[1]), rewardpos)
# select state, action, nextstate, absorbin
sast_ = data[:, indicies]  
actions = (np.arange(3)).tolist()

fqi_list = []

for i in xrange(0,n_cycle): 
    alg = ExtraTreesRegressor(n_estimators=50, criterion='mse',
                                     min_samples_split=2, min_samples_leaf=1)
    
    #print ("batching the data....")
    np.random.shuffle(sast_)  
    sast = sast_[0:data_size,:] 
    #print ("done!")
    states = np.array(sast[:,0:stateDim])
    std_var = np.mean(np.std(states))
    h = std_var* (4./(3.*data_size)) ** 0.2
    
    #old_scores = np.copy(scores)
    
    kde = KernelDensity(kernel="gaussian",bandwidth=h).fit(states)
    
    scores = kde.score_samples(states)
    
    #mean of the scores
    mean_scores.append(np.mean(scores))
    print("mean_scores: ", mean_scores[-1])
    #variance of the scores
    std_scores.append(np.std(scores))
    print("std_scores: ", std_scores[-1])
    #mean of the variance of the states
    std_states.append(np.mean(np.std(states, axis=1)))
    print("std_states: ", std_states[-1])
    #precedence_diff = np.sum((scores-old_scores)**2)
    # select reward
    r =  -scores
    
    #-----------------------------------------------------------------------------
    #Run FQI
    #-----------------------------------------------------------------------------
              
    fqi_list.append(FQI(estimator=alg,
              stateDim=sdim, actionDim=adim,
              discrete_actions=actions,
              #TODO: set scaled again
              gamma=0.99,scaled=False, horizon=31, verbose=1))
    
    fqi = fqi_list[i]
              
    environment = BlackBox()
    
    init_state = environment.getState()
    
    l1=[0]*(n_iter-1)
    l2=[0]*(n_iter-1)
    linf=[0]*(n_iter-1)
    
    new_q = np.zeros((4,data_size))
    for iteration in range(n_iter):
        #fit
        if(iteration==0):
            fqi.partial_fit(sast, r, **fit_params)
        else:
            fqi.partial_fit(None, None, **fit_params)

        mod = fqi.estimator
        
        
        old_q = np.copy(new_q)
        if iteration > 0:
            for action in actions:
                col_action = np.matrix([action]*data_size)
                
                state_action = np.concatenate((states, col_action.T), axis=1)
                
                new_q[action] = mod.predict(state_action)
                
            new_q = np.array(new_q)
            
            l1[iteration-1] = np.mean(np.abs(old_q - new_q))
            l2[iteration-1] = np.sqrt(np.mean((old_q - new_q)**2))
            linf[iteration-1] = np.max(np.abs(old_q - new_q))
            
            
            print("l2: " , l2[iteration-1])
            np.save(sample.path + "/" + "l1_" + str(i), l1)
            np.save(sample.path + "/" + "l2_" + str(i), l2)
            np.save(sample.path + "/" + "linf_" + str(i), linf)
        
        ## test on the simulator
        """environment.reset()
        print ("start running")
        assert(np.array_equal(init_state, environment.getState()))
        
        tupla, rhistory = runEpisode(fqi, environment, 0.99)
        t, j, s = tupla
        print ("score: " + str(t))
        
        #environment.end()
        
        #keep track of the results
        step.append(t) 
        J.append(j)
        #loss.append(rhistory)
        if(s==1):
            goal=1"""
    
    print("running env")
    new_sast = np.array(runEpisodeToDS(environment,fqi_list, eps_out)) 
    environment.reset()    
    print("done!")
    #print("shuffle new_sast")
    np.random.shuffle(new_sast)
    #let's save the file every cycle
    np.save(sample.path + "/sast" + str(i) ,new_sast)
    #print("append a portion of new_sast")
    sast_ = np.append(sast_,new_sast[0:data_size,:],axis=0)
    #print("done!")
    

"""---------------------------------------------------------------------------
Saving phase
---------------------------------------------------------------------------"""

"""
sample.addVariableResults("step",str(step))
sample.addVariableResults("J",J)
#sample.addVariableResult("loss",loss)
sample.addVariableResults("goal",str(goal))

sample.closeSample()
"""