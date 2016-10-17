import ifqi.envs as envs
import ifqi.models as md
from ifqi.utils.datasetGenerator import DatasetGenerator
from ifqi.fqi.FQI import FQI
import ifqi.evaluation.evaluation as evaluate
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesRegressor
from ifqi.models.actionRegressor import ActionRegressor
import matplotlib.pyplot as plt

mdp = envs.CartPole()
stateDim, actionDim = envs.getSpaceInfo(mdp)
regressor_params = {'n_estimators': 50}
discrete_actions = mdp.action_space.values
regressor = ExtraTreesRegressor(**regressor_params)

# md.ActionRegressor(regressor, discreteActions=discrete_actions, decimals=5, **regressor_params)

dataset = DatasetGenerator(mdp)
dataset.generate(n_episodes=100)

fqi = FQI(estimator=regressor,
          stateDim=stateDim,
          actionDim=actionDim,
          discreteActions=discrete_actions,
          gamma=mdp.gamma,
          horizon=mdp.horizon,
          verbose=True,
          features=None,
          scaled=True)

fitParams = {}
# fitParams = {
#     "nEpochs": 300,
#     "batchSize": 50,
#     "validationSplit": 0.1,
#     "verbosity": False,
#     "criterion": "mse"
# }

fqi.partial_fit(*dataset.sastr, **fitParams)

iterations = 10
iterationValues = []
for i in range(iterations - 1):
    fqi.partial_fit(None, None, **fitParams)
    values = evaluate.evaluate_policy(mdp, fqi, nbEpisodes=10)
    iterationValues.append(values[0])

    if i == 1:
        fig1 = plt.figure(1)
        ax = fig1.add_subplot(1, 1, 1)
        h = ax.plot(range(i + 1), iterationValues, 'ro-')
        plt.ylim(min(iterationValues), max(iterationValues))
        plt.xlim(0, i + 1)
        plt.ion()  # turns on interactive mode
        plt.show()
    elif i > 1:
        h[0].set_data(range(i + 1), iterationValues)
        ax.figure.canvas.draw()
        plt.ylim(min(iterationValues), max(iterationValues))
        plt.xlim(0, i + 1)
        # plt.show()
