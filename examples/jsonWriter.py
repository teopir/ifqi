"""
Here we easily organize a json
"""

from __future__ import print_function
import json
import os

def ask(message, err_message, function):
    # type: (string, string, lambda) -> object
    while True:
        try:
            ret = raw_input(message)
            ret = function(ret)
            print("OK!")
            break
        except KeyboardInterrupt:
            print
            'Interrupted'
            exit()
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

def askList(message, myList):
    while True:
        answer = raw_input(message)
        if str(answer) in myList:
            return answer
        print("Type please one of those: " + str(myList))

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

mdpName = askList("Insert the name of the environemnt ", ["CarOnHill","SwingUpPendulum",
"Acrobot", "BicycleBalancing", "BicycleNavigate", "SwingPendulum",
"CartPole", "CartPoleDisc", "LQG1D"])

isList = False
discreteActions = None
fitActions = False

if mdpName in ["LQG1D"]:
    message = "Insert a list of actions (sorrunded with square brackets, separated with commas): "
    while not isinstance(discreteActions, list) or len(discreteActions) < 2:
        discreteActions = input(message)
        message = "Please, insert a list of action of kind: [a1, a2, .. ] where a1, a2 are floats and with len>=2: "
    print("OK!")
    fitActions = True

print("-"*nLine + "\nExperimentSetting\n" + "-"*nLine)

jsonFile = {}

jsonFile["experimentSetting"] = {
        "datasets":ask("Select the number of datasets ", "Number of dataset should be integer and positive ", positiveNumber),
        "nRepetitions":ask("Select the number of repetition on each dataset ", "Number of dataset should be integer and positive ", positiveNumber),
        "evaluations":{
            "everyNIterations":ask("Select how often would you like to evaluate the policy found ", "Please insert a positive number ", positiveNumber),
            "nEvaluations":ask("Select how many episodes would you like to run on your environment ", "Please insert a positive integer number ", positiveNumber)
        }
    }

newSize = True
sizes = []
while newSize:

    size = ask("Insert a size (number of episodes) for the experiment ","Insert a positive integer number please ", positiveNumber)
    sizes.append(size)
    if not ask("Would you like to insert another size? ", "Just type 'y' to say Yes ",yesQuestion):
        newSize = False

jsonFile["experimentSetting"]["sizes"] = sizes

print("-"*nLine + "\nRegressors Section\n" + "-"*nLine)

newRegressor = True
regressorList = []
while newRegressor:
    regressorName = askList('Insert the name of the regressor you wish: ', ["ExtraTree",
"ExtraTreeEnsemble","Linear","LinearEnsemble","MLP","MLPEnsemble"])

    if regressorName == "MLP" or regressorName == "MLPEnsemble":
        regressor = {
            "modelName":regressorName,
            "nHiddenNeurons":ask("Define the number of hidden neurons you wish to have: ", "Insert a positive numer ", positiveNumber),
            "nLayers":ask("Define the number of hidden layers you wish to have: ", "Insert a positive numer ", positiveNumber),
            "optimizer":ask("Which optimizer would you like to have: ", "Insert a valid string ", str),
            "activation":ask("Which activation function would you like to have: ", "Insert a valid string ", str),
            "supervisedAlgorithm":{
                "nEpochs":ask("How many epochs would you like? ", "Positive integer required ", yesQuestion),
                "batchSize":ask("What's the batch size? ", "Positive integer required ", yesQuestion),
                "validationSplit":ask("What's the validation split (from 0 to 1)? ", "Float between 0 e 1 ", float ),
                "verbosity":ask("Would you like Supervised Algorithm to be verbose(Y|n)? ", "Just type 'y' for yes ", yesQuestion),
                "criterion":"mse"
            }
        }
    elif regressorName == "ExtraTree" or regressorName == "ExtraTreeEnsemble":
        regressor = {
            "modelName": regressorName,
            "nEstimators": ask("Define the number of trees you would like to have: ", "Insert a positive numer ", positiveNumber),
            "minSamplesSplit": ask("Define minSamplesSplit: ", "Insert a positive numer ", positiveNumber),
            "minSamplesLeaf": ask("Define minSamplesLeaf: ", "Insert a positive numer ", positiveNumber),
            "supervisedAlgorithm": {
                "criterion": "mse"
            }
        }
    elif regressorName == "Linear" or regressorName == "LinearEnsemble":
        regressor =  {
            "model_name": regressorName,
            "features": {
                "name": "poly",
                "degree": ask("Define the degree of the polinomial features: ", "Insert a positive numer ", positiveNumber)
            },
            "supervisedAlgorithm": {
                "criterion": "mse"
            }
        }
    else:
        raise Exception("The regressor name is not defined")

    if not raw_input("Would you like to insert another regerssor?").capitalize() in ["Y", "YE", "YES", 1, "1"]:
        newRegressor = False

    regressor["fitActions"] = fitActions

    regressorList.append(regressor)



print("-"*nLine + "\nRLAlgorithm Section - FQI Default\n" + "-"*nLine )
lrAlgorithm = {
        "algorithm":"FQI",
        "nIterations":ask("Prompt the number of iterations of your desire: ", "Please insert a positive integer number ", positiveNumber),
        "gamma":ask("What's the gamma? ","Insert a float (between 0 and 1) ", float),
        "verbosity":ask("Would you like that FQI will be verbose (Y|n)? ","Say Y to say yes ", yesQuestion),
        "horizon":ask("Insert the horizon", "Insert a positive integer number ", yesQuestion),
        "scaled":ask("Would you like to scale the dataset? (Y|n) ","To say yes just type y ", yesQuestion)
    }

if ask("Would you like to have experience replay (Y|n) ", "Say y to say yes ", yesQuestion):
    lrAlgorithm["experienceReplay"] = {
            "everyNIterations":ask("how many iterations would you like to collect experience? ","Please insert positive integer ", positiveNumber),
            "nExperience":ask("How many episodes would you like to collect? ", "Please insert a positive integer ", positiveNumber)
        }

jsonFile["regressors"] = regressorList
jsonFile["rlAlgorithm"] = lrAlgorithm

print("-"*nLine + "\nSupervised Algorithm Session\n" + "-"*nLine)

#TODO: move inside MLP in regressors
#jsonFile["supervisedAlgorithm"] =

jsonFile["mdp"] = {
        "mdpName":mdpName,
        "discreteActions":discreteActions
    }

jsonFile["version"]=1

fileName =  ask("What is the name of your new json?" , "Just type a string", str)

print("-"*nLine + "\nHere Your file\n" + "-"*nLine)
print (jsonFile)

directory = os.path.dirname(fileName)
if not os.path.isdir(directory): os.makedirs(directory)
with open(fileName, 'w') as fp:
    json.dump(jsonFile, fp)
