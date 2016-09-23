from __future__ import print_function
import os
import sys
import cPickle
import numpy as np

sys.path.insert(0, os.path.abspath('../'))


"""

Single thread of experimentThreadManager
"""


from ifqi.experiment import Experiment
from ifqi.fqi.FQI import FQI
from ifqi.utils.parser import loadReLeDataset
from ifqi.utils.datasetCollector import loadIfqiDataset

"""
save the variable
dir_ : path of the folder
d : number of the dataset
e : number of the experiment
t : number of the iteration
var_anme: name of the variable to save
var: content of the variable
"""
def saveVariable(dir_,d,e,t,var_name,var):
    directory =os.path.dirname(dir_+ "/" + var_name + "_" +  d + "_" + str(e) + "_" + str(t) + ".npy")
    if not os.path.isdir(directory): os.makedirs(directory)
    np.save(dir_+ "/" + var_name +"_"+  d + "_" + str(e) + "_" + str(t), var)
    
    
#------------------------------------------------------------------------------
# Retrive params
#------------------------------------------------------------------------------
config_file = sys.argv[1]
d = sys.argv[2]
e = int(sys.argv[3])


exp = Experiment(config_file)

if 'MLP' in exp.config['model']['modelName']:
    fit_params = {'nb_epoch': exp.config['supervisedAlgorithm']['nEpochs'],
                  'batch_size': exp.config['supervisedAlgorithm']['batchSize'],
                  'validation_split': exp.config['supervisedAlgorithm']['validationSplit'],
                  'verbose': exp.config['supervisedAlgorithm']['verbosity']
                  }
else:
    fit_params = dict()
    
data, state_dim, action_dim, reward_dim = loadReLeDataset(
        '../dataset/' + exp.config['experimentSetting']['loadPath'] + "/"+ d + '.txt')
assert(state_dim == exp.mdp.stateDim)
assert(action_dim == exp.mdp.actionDim)
assert(reward_dim == 1)

rewardpos = state_dim + action_dim
indicies = np.delete(np.arange(data.shape[1]), rewardpos)

sast = data[:, indicies]
r = data[:, rewardpos]


print('Experiment: ' + str(e))
        
exp.loadModel()

if 'features' in exp.config['model']:
    features = exp.config['model']['features']
else:
    features = None

dir_ = "results/" + exp.config["experimentSetting"]["savePath"]

#-----------------------------------------------------------------------------
# FQI Loading
#-----------------------------------------------------------------------------

if not os.path.isfile(dir_ + "/" +  d + "_" + str(e) + ".pkl"):    #if no fqi present in the directory
    fqi = FQI(estimator=exp.model,
          stateDim=state_dim,
          actionDim=action_dim,
          discreteActions=exp.mdp.getDiscreteActions(),
          gamma=exp.config['rlAlgorithm']['gamma'],
          horizon=exp.config['rlAlgorithm']['horizon'],
          verbose=exp.config['rlAlgorithm']['verbosity'],
          features=features,
          scaled=exp.config['rlAlgorithm']['scaled'])
    fqi.partial_fit(sast[:], r[:], **fit_params)
    min_t = 1
else:
    fqi_obj = cPickle.load(open(dir_ + "/" + d + "_" + str(e) +  ".pkl", "rb"))
    fqi = fqi_obj["fqi"]
    min_t = fqi_obj["t"] + 1

#------------------------------------------------------------------------------
# FQI Iterations
#------------------------------------------------------------------------------


replay_experience = False
for t in range(min_t, exp.config['rlAlgorithm']['nIterations']):
    
    # Partial fit 
    if replay_experience:
        fqi.partial_fit(sast[:], r[:], **fit_params)
        replay_experience=False
    else:
        fqi.partial_fit(None, None, **fit_params)
        
    
    if "saveIteration"  in exp.config['experimentSetting']:
        
        
        if t % exp.config['experimentSetting']["saveIteration"] == 0:
            print("Start evaluation")
            
            score, step, goal = exp.mdp.evaluate(fqi, expReplay=False, n_episodes=exp.config['experimentSetting']['nRepeatEpisodes'])
            
            saveVariable(dir_, d, e, t, "score", score)
            saveVariable(dir_, d, e, t, "step", step)
            saveVariable(dir_, d, e, t, "goal", goal)
            
            print("End evaluation")
            
    #--------------------------------------------------------------------------
    # SAVE FQI STATUS
    #--------------------------------------------------------------------------
    if t % exp.config['experimentSetting']["saveFqi"] == 0:
        directory =os.path.dirname(dir_ + "/" + d + "_" + str(e) + ".pkl")
        if not os.path.isdir(directory): os.makedirs(directory)
        cPickle.dump({'fqi':fqi,'t':t},open(dir_ + "/" + d + "_" + str(e) + ".pkl", "wb"))
        
    #--------------------------------------------------------------------------
    # Experience Replay
    #--------------------------------------------------------------------------
        
    if "experienceReplay" in exp.config['experimentSetting']:
        if t % exp.config['experimentSetting']["experienceReplay"] == 0:
            print ("init experience replay")
            for _ in xrange(exp.config['experimentSetting']["nReplay"]):
                
                
                score, step, goal, sast_temp, r_temp = exp.mdp.evaluate(fqi, expReplay=True)
            
                np.concatenate((sast, sast_temp),axis=0)
                np.concatenate((r, r_temp),axis=0)
                
                indx = np.array(range(r.shape[0]))
                np.random.shuffle(indx)
                
                sast = sast[indx,:]
                r = r[indx]
                
            replay_experience = True
            if exp.config["rlAlgorithm"]["resetFQI"]:
                fqi.iteration=0
            print ("end experience replay")
            
#------------------------------------------------------------------------------
# Save at the end
#------------------------------------------------------------------------------
            
print("Start evaluation")
score, step, goal = exp.mdp.evaluate(fqi,expReplay=False)
print("End evaluation")

saveVariable(dir_, d, e, "last_", "score", score)
saveVariable(dir_, d, e, "last_", "step", step)
saveVariable(dir_, d, e, "last_", "goal", goal)
