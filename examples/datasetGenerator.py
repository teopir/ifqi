"""bicycleDatasetGenerator.py

This source generate dataset for the bycicle
"""
import numpy as np


def runBicycleEpisode(environment, o_file, n_action,  t_max):
    fout = o_file
    environment.reset()
    interrupted = False
    t = 0
    while(not environment.isAbsorbing()):
        state = environment.getState()
        action = np.random.randint(n_action)
        r = environment.step(action)
        # 1 0 interrotto
        # 1 1 episodio finito
        # 0 0 ... episodio non interrotto non finito
        fout.write("0,0,")
        for s in state:
            fout.write(str(s)+",")
        fout.write(str(action)+",")
        fout.write(str(r)+"\n")
        if(t>=t_max):
            interrupted = True
            break
        t+=1
    
    if interrupted:
        fout.write("1,0,")
    else:
        fout.write("1,1,")
    state = environment.getState()
    for i in range(0, len(state)-1):
        fout.write(str(state[i])+",")
    fout.write(str(state[-1])+"\n")

def generateDataset(environment, folder_name, state_dim, action_dim, reward_dim, n_action, dataset_number, episodes_number, max_episode_length=100):
    for i in range(dataset_number):
        fout = open("dataset/" + folder_name + str(i) +".txt" , "w")
        fout.write(str(state_dim) + "," + str(action_dim) + "," + str(reward_dim) +"\n")
        for _ in range(episodes_number):
            runBicycleEpisode(environment, fout, n_action, max_episode_length)
        fout.close()

