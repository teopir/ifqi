# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 17:38:03 2016

@author: samuele

Porting to json

Example on usage

python jsonWriter.py json_path=linear_thread.json load_path="bicycle_data" 
    model_name="Linear" dataset_folders=[size_500/bicycle,size_1000/bicycle,size_1500/bicycle] 
    save_path=bicycle_balanced_linear_thread n_dataset=10 save_iteration=1 n_experiments=1 n_iterations=40 
    unsupervised_verbosity=1 mdp_name=BicycleBalancing scaled=1 degree=5
    
p.s is not necessary to use all the parameters :) if not specified, the parameter will be set to its default value

"""
import sys
import json
import os

"""this allow to read another json file and "copy" the same setting"""

def retreiveParamsFromFile(jsonPath):
    with open(jsonPath) as data_file:    
        data = json.load(data_file)
        
    for key, val in data.items():
        if key=="supervised_algorithm":
            for k, v in val.items():
                if key=="verbose":
                    globals()["supervised_verbose"] = v
                else:
                    globals()[k] = v
        elif key=="rl_algorithm":
            for k, v in val.items():
                if key=="verbose":
                    globals()["unsupervised_verbose"] = v
                else:
                    globals()[k] = v
        else:
            for k, v in val.items():
                globals()["unsupervised_verbose"] = v

    
    return data
    
def retreiveParams():
    global folder,test,case,code, number, library
    commands = sys.argv[1:]
        
    for arg in commands:
        s = arg.split("=")    
        name = s[0]
        value = s[1]
        if name=="copy_from":
            retreiveParamsFromFile(value)
        else:
            
            if("[" in value):
                value = value.replace("[", "")
                value = value.replace("]","")
                value = value.replace(" ","")
                value = value.split(",")
            
            try:
                if type(value) is list:
                    globals()[name] = value
                elif "." in value:
                    globals()[name] = float(value)
                else:
                    globals()[name] = int(value)
            except:
                globals()[name] = str(value)
            
json_path="experiment.json"
name="exp1"
load_path = ""
save_path = ""
degree = 4
n_dataset = 20
n_experiments = 1
model_name = "mlp"
n_hidden_neurons = 10
n_layers = 2
optimizer= "rmsprop"
activation="relu"
mdp_name="MountainCar"
n_epoch=300
save_fqi = 5
batch_size=100
validation_split=0.1
supervised_verbosity=0
unsupervised_verbosity=0
n_iterations=20
gamma=0.98
horizon = 50000
scaled = 1
dataset_folders = []
n_estimators = 50
min_samples_split=16
min_samples_leaf=8
criterion="mse"
save_iteration=1000
nThread = 4
experience_replay=10
n_replay=100
add_last=False

retreiveParams()

datasets = list(folder+str(num) for folder in dataset_folders for num in range(0,n_dataset))

def camelize(dic, key=False):
    if isinstance(dic,dict):
        ret = {}
        for key in dic:
            ret[camelize(key, key=True)] = camelize(dic[key])
        return ret
    elif key:
        ret = ""
        cam = False
        for letter in dic:
            if letter=="_":
                cam=True
            elif cam:
                ret+=letter.upper()
                cam=False
            else:
                ret+=letter
        return ret
    else:
        return dic
        

json_file = {
    "experiment": {
        "name":name,
        "date":"..."
    },
    "experiment_setting":{
        "load_path":load_path ,
        "save_path":save_path,
        "datasets":datasets,
        "n_experiments":n_experiments,
        "save_iteration":save_iteration,
        "save_fqi":save_fqi,
        "experience_replay":experience_replay,
        "n_replay":n_replay
    },
    "model":{
        "model_name":model_name,
        "n_hidden_neurons":n_hidden_neurons,
        "n_layers":n_layers,
        "optimizer":optimizer,
        "activation":activation
    },
    "mdp":{
        "mdp_name":mdp_name    
    },
    "supervised_algorithm":{
        "n_epochs":n_epoch,
        "batch_size":batch_size,
        "validation_split":validation_split,
        "verbosity":supervised_verbosity
    },
    "rl_algorithm":{
        "algorithm":"FQI",
        "n_iterations":n_iterations,
        "gamma":gamma,      
        "verbosity":unsupervised_verbosity,
        "horizon":horizon,
        "scaled":scaled
    }
}
if model_name == "ExtraTree" or model_name == "ExtraTreeEnsemble":
    json_file["model"] = {
        "model_name": model_name,
        "n_estimators": n_estimators,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf
    }
    json_file["supervised_algorithm"] ={
        "criterion":criterion
    }
if model_name == "ExtraTree" or model_name == "ExtraTreeEnsemble":
    json_file["model"] = {
        "model_name": model_name,
        "n_estimators": n_estimators,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf
    }
    json_file["supervised_algorithm"] ={
        "criterion":criterion    
    }
if model_name == "Linear" or model_name == "LinearEnsemble":
    json_file["model"] = {
        "model_name": model_name,
        "features": {
            "name": "poly",
            "degree": degree
        }
    }
    json_file["supervised_algorithm"] ={
        "criterion":criterion    
    }
    
with open("results/" + save_path + ".json", 'w') as fp:
    json.dump(camelize(json_file), fp)

cmd = "python experimentThreadManager.py " + save_path + ".json" + " " +  str(nThread) + " " + str(add_last)
os.system(cmd)
