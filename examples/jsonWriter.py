"""
Here we easily organize a json
"""

from __future__ import print_function
import json

def ask(message, err_message, function):
    # type: (string, string, lambda) -> object
    while True:
        try:
            ret = raw_input(message)
            ret = function(ret)
            print("OK!")
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

def getRegressor(name, errorMessage=""):

    if (name == 'ExtraTree' 
        or name == 'ExtraTreeEnsemble'
        or name == 'MLP'
        or name == "MLPEnsemble"
        or name == "Linear"
        or name == "LinearEnsemble"):
            return name
    else:
        raise Exception(errorMessage)

yesQuestion = lambda x: str(x).capitalize() in ["Y","YE","YES",1,"1"]

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
        "nExperiments":ask("Select the number of repetition on each dataset ", "Number of dataset should be integer and positive", positiveNumber),
        "evaluations":{
            "everyNIterations":ask("Select how often would you like to evaluate the policy found", "Please insert a positive number", positiveNumber),
            "nEvaluations":ask("Select how many episodes would you like to run on your environment", "Please insert a positive integer number", positiveNumber)
        }
    }

newSize = True
sizes = []
while newSize:

    size = ask("Insert a size (number of episodes) for the experiment","Insert a positive integer number please", positiveNumber)
    sizes.append(size)
    if ask("Would you like to insert another size?", "Just type 'y' to say Yes",yesQuestion):
        newSize = False

jsonFile["experimentSetting"]["sizes"] = sizes

newRegressor = True
regressorList = []
while newRegressor:
    regressorName = getRegressor('Insert the name of the regressor you wish: ', """Please insert a name from the following list
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
            "minSamplesSplit": ask("Define minSamplesSplit: ", "Insert a positive numer", positiveNumber),
            "minSamplesLeaf": ask("Define minSamplesLeaf: ", "Insert a positive numer", positiveNumber)
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

    if not raw_input("Would you like to insert another regerssor?").capitalize() in ["Y", "YE", "YES", 1, "1"]:
        newRegressor = False

    regressorList.append(regressor)



print("-"*nLine + "RLAlgorithm Section - FQI Default" + "-"*nLine )
lrAlgorithm = {
        "algorithm":"FQI",
        "nIterations":ask("Prompt the number of iterations of your desire: ", "Please insert a positive integer number", positiveNumber),
        "gamma":ask("What's the gamma? ","Insert a float (between 0 and 1)", float),
        "verbosity":ask("Would you like that FQI will be verbose (Y|n)?","Say Y to say yes", yesQuestion),
        "horizon":ask("Insert the horizon", "Insert a positive integer number ", yesQuestion),
        "scaled":ask("Would you like to scale the dataset? (Y|n)","To say yes just type y", yesQuestion)
    }

if ask("Would you like to have experience replay (Y|n)", "Say y to say yes", yesQuestion):
    lrAlgorithm["replayExperience"] = {
            "everyNIterations":ask("how many iterations would you like to collect experience?","Please insert positive integer", positiveNumber),
            "nExperience":ask("How many episodes would you like to collect?", "Please insert a positive integer", positiveNumber)
        }

jsonFile["regressors"] = regressorList
jsonFile["rlAlgorithm"] = lrAlgorithm

print("-"*nLine + "\nSupervised Algorithm Session\n" + "-"*nLine)


jsonFile["supervisedAlgorithm"] = {
        "nEpochs":ask("How many epochs would you like?", "Positive integer required", yesQuestion),
        "batchSize":ask("What's the batch size?", "Positive integer required", yesQuestion),
        "validationSplit":ask("What's the validation split (from 0 to 1)?", "Float between 0 e 1", float ),
        "verbosity":ask("Would you like Supervised Algorithm to be verbose(Y|n)?", "Just type 'y' for yes", yesQuestion)
    }

jsonFile["mdp"] = {
        "mdp_name":mdpName
    }

fileName =  ask("What is the name of your new json?" , "Just type a string", str)
with open("results/" + fileName + ".json", 'w') as fp:
    json.dump(camelize(json_file), fp)