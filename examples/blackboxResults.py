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

def runAction(environment, action,t,horizon, n=100):
    i = 0
    r = 0
    while t<horizon and environment.next:
        r += environment.step(action)
        i+=1
        if i>=n:
            break
    return r
    
def runOnTram(myfqi, environment, eps, sast,t,r, horizont=10000):
    rnd = 0 #np.random.rand()
    while t<horizont and environment.next and eps>rnd:
        #print("tram started" , t)
        state = environment.getState()
        action, _ = myfqi.predict(np.array(state))
        #r.append(environment.step(action))
        r.append(runAction(environment, action,t, horizont))
        next_state =  environment.getState()
        sast.append(state.tolist() + [action] + next_state.tolist() + [1-environment.next])
        rnd = np.random.rand()
        t+=1
        
        #print("tram finished" , t)
    return sast, t, r
    
def runEpisodeToDS(environment, fqi_list,eps_out,horizont, eps_on_tram=0.01):   # (nstep, J, success)

    sast = []
    r = []
    t=0
    while t<horizont and environment.next:
        state = environment.getState()
        #action, _ = myfqi.predict(np.array(state))
        rnd = np.random.rand()
        if rnd > eps_on_tram:
            action = np.random.randint(4)
        else:
            action = np.random.randint(len(fqi_list))
            len_before = len(sast)
            sast, t, r = runOnTram(fqi_list[action], environment, eps_out, sast,t,r, horizont)
            len_after =  len(sast)
            assert(len_after>len_before)
            continue
        #r.append(environment.step(action))
        r.append(runAction(environment, action,t, horizont))
        next_state =  environment.getState()
        sast.append(state.tolist() + [action] + next_state.tolist() + [1-environment.next])
        t+=1
    return sast,r

       
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
min_split = 100
min_leaf = 36


#plotting configuration
last_layer = False
qfunction = False
policy = False

#variable to save
std_scores = []
std_states = []
mean_scores = []
min_range = []
max_range = []

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
                                     min_samples_split=min_split, min_samples_leaf=min_leaf)
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



rewardpos = sdim + adim
stateDim = sdim
indicies = np.delete(np.arange(data.shape[1]), rewardpos)
# select state, action, nextstate, absorbin
sast_ = data[:, indicies]  
actions = (np.arange(4)).tolist()

np.random.shuffle(sast_)  
sast = sast_[0:,:] 

all_states = sast[:,0:stateDim]
std_all_states = np.mean(np.std(all_states,axis=0))
h = std_all_states* (4./(3.*data_size)) ** 0.2
kde_ = KernelDensity(kernel="gaussian",bandwidth=h).fit(all_states)
fqi_list = []

#TODO: n_cycle
for i in xrange(0,n_cycle): 
    alg = ExtraTreesRegressor(n_estimators=10, criterion='mse',
                                     min_samples_split=8, min_samples_leaf=4)
    
    #print ("batching the data....")
    #np.random.shuffle(sast_)  
    #sast = sast_[0:,:] 
    
    #print ("done!")
        
    scores = kde_.score_samples(all_states)
    
    # select reward
    r =  -scores
    
    #-----------------------------------------------------------------------------
    #Run FQI
    #-----------------------------------------------------------------------------
              
    fqi_list.append(FQI(estimator=alg,
              stateDim=sdim, actionDim=adim,
              discrete_actions=actions,
              #TODO: set scaled again
              gamma=0.9,scaled=scaled, horizon=31, verbose=1))
    
    fqi = fqi_list[i]
              
    environment = BlackBox()
    
    init_state = environment.getState()
    
    l1=[0]*(n_iter-1)
    l2=[0]*(n_iter-1)
    linf=[0]*(n_iter-1)
    
    sast_dim = sast.shape[0]
    new_q = np.zeros((4,sast_dim))
    #TODO:remove
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
                col_action = np.matrix([action]*sast_dim)
                
                state_action = np.copy(np.concatenate((all_states, col_action.T), axis=1))
                if scaled:
                    state_action = fqi._sa_scaler.transform(state_action)
                
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
    temp_sast, r = runEpisodeToDS(environment,fqi_list, eps_out,data_size*(i+1))
    
    print("ho esplorato fino a: " + str(environment.get_time()))
    new_sast = np.array(temp_sast) 
    print("new_sast: ", new_sast.shape)
    environment.reset()    
    print("done!")
    
    
    #print("shuffle new_sast")
    np.random.shuffle(new_sast)
    #let's save the file every cycle
    """np.save(sample.path + "/sast" + str(i) ,new_sast)    
    np.save(sample.path + "/sastReward" + str(i) ,r)"""
    #print("append a portion of new_sast")
    
    #JUST TAKE A BATCH
    #new_sast = np.copy(new_sast[0:data_size,:])
    
    
    #sast collect data_size from new_sast. so it has the history. then we will again take a batch from it 
    sast = np.append(sast,np.copy(new_sast[:,:]),axis=0)
    
    #Just take a piece---------------------------------------------------------
    #np.random.shuffle(sast)
    #batch_size = int(sast.shape[0] * 0.5)
    #sast = sast[:batch_size,:]
    #Faster computation--------------------------------------------------------

    
    #all the states of the history
    all_states = sast[:,0:stateDim]
    
    print("kernel computing")
    
    std_all_states = np.mean(np.std(all_states,axis=0))
    h = std_all_states* (4./(3.*data_size)) ** 0.2
    
    kde_ = KernelDensity(kernel="gaussian",bandwidth=h).fit(all_states)
    all_scores = kde_.score_samples(all_states)
    
    std_all_scores = np.std(all_scores)
    mean_all_scores = np.mean(all_scores)
    print("done!")
    
    if(i!=0):
        old_last_states_min = np.copy(states_min)
        old_last_states_max = np.copy(states_max)
    
    print("variable computing:")
    #just a batch of the last episode's states
    last_states = np.copy(new_sast[:,0:stateDim])

    #free some memory
    del new_sast
    gc.collect()
    
    #computing the ranges max and min of all the set
    states_max = np.max(all_states,axis= 0)
    states_min = np.min(all_states,axis= 0)
    last_scores = kde_.score_samples(last_states)
    
    if(i!=0):
        #quadratic error on previous range of states: 
        #-----sqrt of mean of square
        #-----like all others mesures it is inaccurate because it involves just a batch of the data
        min_range.append(np.sqrt(
                        np.mean( 
                                (old_last_states_min-states_min)**2 
                                )))
        max_range.append(np.sqrt(
                        np.mean( 
                                (old_last_states_max-states_max)**2 
                                )))
        print("Difference between two mins: ", min_range)
        print("Difference between two maxs: ", max_range)
    
    #let's save the relatives data
    mean_scores.append(np.mean(last_scores)/mean_all_scores)
    print("mean_scores: ", mean_scores[-1])
    #variance of the scoresp.
    std_scores.append(np.std(last_scores)/std_all_scores)
    print("std_scores: ", std_scores[-1])
    #mean of the variance of the states
    std_states.append(np.mean(np.std(last_states, axis=0))/std_all_states)
    print("std_states: ", std_states[-1])
    print("end cycle")
    #TODO: printare qua le variabili
    #print("done!")
    
    print ("Run Episode and save it")
    temp_sast, t, r = runOnTram(fqi_list[-1],environment,1,[],0,[], horizont=13000000)
    #runOnTram(myfqi, environment, eps, sast,t,r, horizont=10000):
    
    np.save(sample.path + "/fqidata" + str(i), np.array(temp_sast))
    np.save(sample.path + "/rewards" + str(i), r)
    print ("done")
    

"""---------------------------------------------------------------------------
Saving phase
---------------------------------------------------------------------------"""



sample.addVariableResults("std_scores", np.array(std_scores))
sample.addVariableResults("std_states", np.array(std_states))
sample.addVariableResults("mean_scores", np.array(mean_scores))
sample.addVariableResults("min_range", np.array(min_range))
sample.addVariableResults("max_range", np.array(max_range))

sample.plotVariable("std_scores")
sample.plotVariable("std_states")
sample.plotVariable("mean_scores")
sample.plotVariable("min_range")
sample.plotVariable("max_range")

#running the last episode
#saving the data so that we can use normal FQI then

print("Running the last experiment")
temp_sast, t, r = runOnTram(fqi_list[-1],environment,1,[],0,[], horizont=13000000)
#runOnTram(myfqi, environment, eps, sast,t,r, horizont=10000):

np.save(sample.path + "/fqidata", np.array(temp_sast))
np.save(sample.path + "/rewards", r)

"""
sample.addVariableResults("")
sample.addVariableResults("step",str(step))
sample.addVariableResults("J",J)
#sample.addVariableResult("loss",loss)
sample.addVariableResults("goal",str(goal))

sample.closeSample()
"""
