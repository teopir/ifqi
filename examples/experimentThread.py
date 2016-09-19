from __future__ import print_function
import os
import sys
import cPickle
import numpy as np

from context import *


"""

Single thread of experimentThreadManager
"""


from ifqi.experiment import Experiment
from ifqi.fqi.FQI import FQI
import ifqi.utils.parser as parser

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

if 'MLP' in exp.config['model']['model_name']:
    fit_params = {'nb_epoch': exp.config['supervised_algorithm']['n_epochs'],
                  'batch_size': exp.config['supervised_algorithm']['batch_size'],
                  'validation_split': exp.config['supervised_algorithm']['validation_split'],
                  'verbose': exp.config['supervised_algorithm']['verbosity']
                  }
else:
    fit_params = dict()
    
data, state_dim, action_dim, reward_dim = parser.parseReLeDataset(
        '../dataset/' + exp.config['experiment_setting']['load_path'] + "/"+ d + '.txt')
assert(state_dim == exp.mdp.state_dim)
assert(action_dim == exp.mdp.action_dim)
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

dir_ = "results/" + exp.config["experiment_setting"]["save_path"]

#-----------------------------------------------------------------------------
# FQI Loading
#-----------------------------------------------------------------------------

if not os.path.isfile(dir_ + "/" +  d + "_" + str(e) + ".pkl"):    #if no fqi present in the directory
    fqi = FQI(estimator=exp.model,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=range(exp.mdp.n_actions),
          gamma=exp.config['rl_algorithm']['gamma'],
          horizon=exp.config['rl_algorithm']['horizon'],
          verbose=exp.config['rl_algorithm']['verbosity'],
          features=features,
          scaled=exp.config['rl_algorithm']['scaled'])
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
for t in range(min_t, exp.config['rl_algorithm']['n_iterations']):
    
    # Partial fit 
    if replay_experience:
        fqi.partial_fit(sast[:], r[:], **fit_params)
        replay_experience=False
    else:
        fqi.partial_fit(None, None, **fit_params)
        
    
    if "save_iteration"  in exp.config['experiment_setting']:
        
        
        if t % exp.config['experiment_setting']["save_iteration"] == 0:
            print("Start evaluation")
            
            score, step, goal = exp.mdp.evaluate(fqi, expReplay=False)
            
            saveVariable(dir_, d, e, t, "score", score)
            saveVariable(dir_, d, e, t, "step", step)
            saveVariable(dir_, d, e, t, "goal", goal)
            
            print("End evaluation")
            
    #--------------------------------------------------------------------------
    # SAVE FQI STATUS
    #--------------------------------------------------------------------------
    if t % exp.config['experiment_setting']["save_fqi"] == 0:
        directory =os.path.dirname(dir_ + "/" + d + "_" + str(e) + ".pkl")
        if not os.path.isdir(directory): os.makedirs(directory)
        cPickle.dump({'fqi':fqi,'t':t},open(dir_ + "/" + d + "_" + str(e) + ".pkl", "wb"))
        
    #--------------------------------------------------------------------------
    # Experience Replay
    #--------------------------------------------------------------------------
        
    if "experience_replay" in exp.config['experiment_setting']:
        if t % exp.config['experiment_setting']["experience_replay"] == 0:
            print ("init experience replay")
            for _ in xrange(exp.config['experiment_setting']["n_replay"]):
                
                score, step, goal, sast_temp, r_temp = exp.mdp.evaluate(fqi, expReplay=True)
            
                np.concatenate((sast, sast_temp),axis=0)
                np.concatenate((r, r_temp),axis=0)
                
                indx = np.array(range(r.shape[0]))
                np.random.shuffle(indx)
                
                sast = sast[indx,:]
                r = r[indx]
                
                replay_experience = True
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
