"""
Here we easily organize a json
"""

from __future__ import print_function

def ask(message, err_message, function):
    while True:
        try:
            ret = raw_input(message)
            ret = function(ret)
            break
        except:
            print(err_message)
    return ret
    
def positiveNumber(x):
    x = int(x)
    if x > 0:
        return x
    raise Exception("Number should be positive")
    
def getMDP(name):
    if (name == 'CarOnHill' 
        or name == 'SwingUpPendulum'
        or name == 'Acrobot'
        or name == "BicycleBalancing"
        or name == "BicycleNavigate"
        or name == "SwingPendulum"
        or name == "CartPole"
        or name == "CartPoleDisc"
        or name == "LQG1D"):
            return name
    else:
        raise Exception("Insert a valid Environment name")

def getRegressor(name):
    if (name == 'ExtraTree' 
        or name == 'ExtraTreeEnsemble'
        or name == 'MLP'
        or name == "MLPEnsemble"
        or name == "Linear"
        or name == "LinearEnsemble"):
            return name
    else:
        raise Exception("Insert a valid Environment name")
        
nLine=10

print("Welcome in JsonWriter 1.0")

mdpName = ask("Insert the name of the environemnt ", """Please select a valid name between: 
\tCarOnHill
\tSwingUpPendulum
\tAcrobot
\tBicycleBalancing
\tBicycleNavigate
\tSwingPendulum
\tCartPole
\tCartPoleDisc
\tLQG1D
""", getMDP)

print("-"*nLine + "\nExperimentSetting\n" + "-"*nLine)

jsonFile = {}

jsonFile["experimentSetting"] = {
        "loadPath":ask("Select the load path ", "loadPath should be a string", str ),
        "savePath":ask("Select the load path ", "loadPath should be a string", str ),
        "datasets":ask("Select the number of datasets ", "Number of dataset should be integer and positive", positiveNumber),
        "nExperiments":ask("Select the number of repetition on each dataset ", "Number of dataset should be integer and positive", positiveNumber)
    }

newRegressor = True
regressorList = []
while newRegressor:
    regressorName = getRegressor("Insert the name of the regressor you wish: ", """Please insert a name from the following list
\tExtraTree
\tExtraTreeEnsemble
\tLinear
\tLinearEnsemble
\tMLP
\tMLPEnsemble    
""")
    if regressorName == "MLP" or regressorName == "MLPEnsemble":
        regressor = {
            "modelName":regressorName,
            "nHiddenNeurons":ask("Define the number of hidden neurons you wish to have: ", "Insert a positive numer", positiveNumber),
            "nLayers":ask("Define the number of hidden layers you wish to have: ", "Insert a positive numer", positiveNumber),
            "optimizer":ask("Which optimizer would you like to have: ", "Insert a valid string", str),
            "activation":ask("Which activation function would you like to have: ", "Insert a valid string", str)
        }
    elif regressorName == "ExtraTree" or regressorName == "ExtraTreeEnsemble":
        regressor = {
            "modelName": regressorName,
            "nEstimators": ask("Define the number of trees you would like to have: ", "Insert a positive numer", positiveNumber),
            "min_samples_split": ask("Define minSamplesSplit: ", "Insert a positive numer", positiveNumber),
            "min_samples_leaf": ask("Define minSamplesLeaf: ", "Insert a positive numer", positiveNumber)
        }
    elif regressorName == "Linear" or regressorName == "LinearEnsemble":
        regressor =  {
            "model_name": regressorName,
            "features": {
                "name": "poly",
                "degree": ask("Define the degree of the polinomial features: ", "Insert a positive numer", positiveNumber)
            }
        }
    else:
        raise Exception("The regressor name is not defined")
    
    regressorList.append(regressor)
