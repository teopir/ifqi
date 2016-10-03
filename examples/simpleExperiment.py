from ifqi.envs.carOnHill import CarOnHill
from ifqi.envs.invertedPendulum import InvPendulum
from ifqi.envs.acrobot import Acrobot
from ifqi.envs.swingPendulum import SwingPendulum
from ifqi.envs.cartPole import CartPole
from ifqi.envs.lqg1d import LQG1D
from sklearn.ensemble import ExtraTreesRegressor
from ifqi.models.mlp import MLP
from sklearn.linear_model import LinearRegression
from ifqi.models.ensemble import ExtraTreeEnsemble, MLPEnsemble, LinearEnsemble
from ifqi.models.actionRegressor import ActionRegressor
from ifqi.utils.datasetGenerator import DatasetGenerator
from gym import spaces
from ifqi.fqi.FQI import FQI
import ifqi.evaluation.evaluation as evaluate
import pylab as plb
import matplotlib.pyplot as plt


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



def askList(message, myList):
    while True:
        answer = raw_input(message)
        if str(answer) in myList:
            return answer
        print("Please, the value should be between: " + str(myList))

environments = {
    "CarOnHill":(CarOnHill,{}),
    "SwingUpPendulum":(InvPendulum,{}),
    "Acrobot":(Acrobot,{}),
    "SwingPendulum":(SwingPendulum,{}),
    "CartPole":(CartPole,{}),
    "CartPoleDisc":(CartPole,{}),
    "LQG1D":(LQG1D,{"discreteRew",True})
}



mdpName = askList("Insert the name of the environemnt ", environments.keys())
environment = environments[mdpName][0](**environments[mdpName][1])

regressors = {
    "ExtraTree":(ExtraTreesRegressor,{'n_estimators': 50,
                      'criterion': 'mse',
                      'min_samples_split': 2,
                      'min_samples_leaf': 5}),
    "ExtraTreeEnsemble":(ExtraTreeEnsemble,{'n_estimators': 50,
                      'criterion': 'mse',
                      'min_samples_split': 2,
                      'min_samples_leaf': 5}),
    "MLP":(MLP,{'nInput': environment.observation_space.shape[0],
                      'nOutput': 1,
                      'hiddenNeurons': 20,
                      'nLayers': 2,
                      'optimizer': 'MRSProp',
                      'activation': 'tanh'}),
    "MLPEnsemble":(MLPEnsemble,{'nInput': environment.observation_space.shape[0],
                      'nOutput': 1,
                      'hiddenNeurons': 20,
                      'nLayers': 2,
                      'optimizer': 'MRSProp',
                      'activation': 'tanh'}),
    "Linear":(LinearRegression,{}),
    "LinearEnsemble":(LinearEnsemble,{})
}


regressorName = askList('Insert the name of the regressor you wish: ', regressors.keys())

if isinstance(environment.action_space, spaces.Box):
    regressor = regressors[regressorName][0](regressors[regressorName][1])
else:
    regressor =  ActionRegressor(regressors[regressorName][0],
                           environment.action_space.values, decimals=5,
                           **regressors[regressorName][1])


size = ask("Insert a size (number of episodes) for the experiment ","Insert a positive integer number please ", positiveNumber)

yesQuestion = lambda x: str(x).capitalize() in ["Y","YE","YES",1,"1"]

if isinstance(environment.action_space, spaces.Box):
    isList = False
    discreteActions = None
    message = "Insert a list of actions (sorrunded with square brackets, separated with commas): "
    while not isinstance(discreteActions, list) or len(discreteActions) < 2:
        discreteActions = input(message)
        message = "Please, insert a list of action of kind: [a1, a2, .. ] where a1, a2 are floats and with len>=2: "
    print("OK!")
else:
    discreteActions = environment.action_space.values

features = None
if regressorName=="Linear" or regressorName=="LinearEnsemble":
    features = {
        "name": "poly",
        "degree": 5
    }

fitParams = {}
if regressorName=="MLP" or regressorName=="MLPEnsemble":
    fitParams = {
        "nEpochs":300,
        "batchSize":50,
        "validationSplit":0.1,
        "verbosity":False,
        "criterion":"mse"
    }

dataset = DatasetGenerator(environment)
dataset.generate(n_episodes=size)

fqi = FQI(estimator=regressor,
          stateDim=environment.observation_space.shape[0],
          #TODO: Fix action dimension
          actionDim=1,
          discreteActions=discreteActions,
          gamma=environment.gamma,
          horizon=environment.horizon,
          verbose=True,
          features=features,
          scaled=True)

iterations = ask("Insert the number of iterations you would like to perform: ","Insert a positive integer number please ", positiveNumber)
fqi.partial_fit(*dataset.sastr,**fitParams)



iterationValues = []
for i in range(iterations):
    fqi.partial_fit(None,None,**fitParams)
    values = evaluate.evaluate_policy(environment,fqi,nbEpisodes=10)
    iterationValues.append(values[0])

    if i==1:
        fig1 = plt.figure(1)
        ax = fig1.add_subplot(1, 1, 1)
        h = ax.plot(range(i+1), iterationValues, 'ro-')
        plt.ylim(min(iterationValues), max(iterationValues))
        plt.xlim(0, i + 1)
        plt.ion()  # turns on interactive mode
        plt.show()
    elif i>1:
        h[0].set_data(range(i+1), iterationValues)
        ax.figure.canvas.draw()
        plt.ylim(min(iterationValues), max(iterationValues))
        plt.xlim(0,i+1)
    #plt.show()
