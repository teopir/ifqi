from __future__ import print_function

import os
import sys
sys.path.append(os.path.abspath('../'))

import unittest
import numpy as np
from ifqi.fqi.FQI import FQI
from ifqi.models.mlp import MLP
from ifqi.models.incr import IncRegression
from ifqi.preprocessors.invPendulumPreprocessor import InvertedPendulumPreprocessor
from ifqi.envs.invertedPendulum import InvPendulum
import ifqi.utils.parser as parser
from sklearn.ensemble import ExtraTreesRegressor
import time

import matplotlib.pyplot as plt
plt.style.use('ggplot')

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
    
def time_to_string(t):
    h = t/3600
    m = (t%3600)/60
    s = (t%3600)%60
    
    s_m = str(m) if len(str(m)) == 2 else "0" + str(m)
    s_s = str(s) if len(str(s)) == 2 else "0" + str(s)
    
    return "ETA: " + str(h) + ":" + s_m + ":" + s_s
    
if __name__ == '__main__':
    
    folders = "ABCDEF"
    
    cls = input ("Do you want to remove previus results? (yes=1, no=0)")
    

    if not os.path.exists("../results/InvPendulum/" + sys.argv[1] + "/" ):
        print("non esiste")
        os.makedirs("../results/InvPendulum/" + sys.argv[1] + "/")
        
    if(cls==1):
        open( "../results/InvPendulum/" + sys.argv[1] +"/results.txt", 'w').close()
        for test in folders:
            open( "../results/InvPendulum/" + sys.argv[1] +"/results" + test + ".txt", 'w').close()
        
        #clear previous file
        
    print("STARTED!!!")
    start = time.time()
    for test in folders:
    
        test_successful = 0
        tot_J = 0
        tot_len = 0
        tot_iter = 0
        for i in range(0,2):
            
            print ("Test: " + test + " n ", i)
            local_start = time.time()
            test_file = "../dataset/pendulum_data/"+test+"/data"+str(i)+".log"
            start = time.time()
            data, sdim, adim, rdim = parser.parseReLeDataset(test_file)
            
            assert(sdim == 2)
            assert(adim == 1)
            assert(rdim == 1)
        
            rewardpos = sdim + adim
            indicies = np.delete(np.arange(data.shape[1]), rewardpos)
        
            # select state, action, nextstate, absorbin
            sast = data[:, indicies]
        
            #prep = InvertedPendulumPreprocessor()
            #sast[:,:3] = prep.preprocess(sast[:,:3])
            
            niter = int(sys.argv[3])
        
            # select reward
            r = data[:, rewardpos]
        
            estimator = sys.argv[1]
            if estimator == 'extra':
                alg = ExtraTreesRegressor(n_estimators=50, criterion='mse',
                                                 min_samples_split=2, min_samples_leaf=1)
                fit_params = dict()
            elif estimator == 'mlp':
                alg = MLP(n_input=sdim+adim, n_output=1, hidden_neurons=5, h_layer=2,
                          optimizer='rmsprop', act_function="sigmoid").getModel()
                fit_params = {'nb_epoch':300, 'batch_size':50, 'verbose':0}
                # it is equivalente to call
                #fqi.fit(sast,r,nb_epoch=12,batch_size=50, verbose=1)
            elif estimator == 'frozen-incr':
                alg = IncRegression(n_input=sdim+adim, n_output=1, hidden_neurons=[5]*(niter+2), 
                          n_h_layer_beginning=2,optimizer='rmsprop', act_function=["sigmoid"]*(niter+2))
                fit_params = {'nb_epoch':300, 'batch_size':50, 'verbose':0}
                # it is equivalente to call
                #fqi.fit(sast,r,nb_epoch=12,batch_size=50, verbose=1)
            elif estimator == 'unfrozen-incr':
                alg = IncRegression(n_input=sdim+adim, n_output=1, hidden_neurons=[5]*(niter+2), 
                          n_h_layer_beginning=2,optimizer='rmsprop', act_function=["sigmoid"]*(niter+2), reLearn=True)
                fit_params = {'nb_epoch':300, 'batch_size':50, 'verbose':0}
            else:
                raise ValueError('Unknown estimator type.')
        
            actions = (np.arange(3)).tolist()
            fqi = FQI(estimator=alg,
                      stateDim=sdim, actionDim=adim,
                      discrete_actions=actions,
                      gamma=0.95, horizon=10, verbose=1)
            #fqi.fit(sast, r, **fit_params)
        
            environment = InvPendulum()
        
            best_tupla = (0,0,0)
            best_iteration = 0
            fout = open("../results/InvPendulum/" + sys.argv[1] + "/results" + test + ".txt", "a")
            
            breakable= (sys.argv[2]=="True")
            this_success=False
            for iteration in range(niter):
                if iteration == 0:
                    fqi.partial_fit(sast, r, **fit_params)
                else:
                    fqi.partial_fit(None, None, **fit_params)

                if(iteration%10==0):
                    mod = fqi.estimator
                    ## test on the simulator
                    environment.reset()
                    tupla, rhistory = runEpisode(fqi, environment, 0.95)
                    t, J, s = tupla
                    if(iteration!=niter-1 and (not (s==1 and breakable))) :
                        fout.write(str(t)+",")
                    else:
                        fout.write(str(t)+"\n")
                    print ("----time: " + str(t))
                    if(t>=best_tupla[0]):
                        best_tupla = tupla
                        best_iteration = iteration
                    if(s==1):
                        if(not this_success):
                            this_success=True
                            test_successful += 1
                        if(breakable):
                            break
            
            tot_iter+=best_iteration +2
                #plt.scatter(np.arange(len(rhistory)), np.array(rhistory))
                #plt.show()
            fout.close()
            print ("Ci ho messo: " + str(time.time() - start) + "s")
            
        mean_J = tot_J/50.
        mean_len = tot_len/50.
        mean_iter = tot_iter/50.
        ETA = int(time.time()-start)    
        s_eta = time_to_string(ETA)
        print ("END TEST: " + s_eta)

        print ("mean_len " + str(mean_len) + " successes: " + str(test_successful) + "/50 " + s_eta)

        #Writing File of results: ------------------------------------

        with open("../results/InvPendulum/" + sys.argv[1] +"/results.txt", "a") as myfile:
            myfile.write("Test Case " + test + "\n")
            myfile.write("J = " + str(mean_J) + ", len = " + str(mean_len) + ", passed = " + str(test_successful) + "/50 "  + "mean_iter: " +str(mean_iter) + " "+ s_eta + "\n")