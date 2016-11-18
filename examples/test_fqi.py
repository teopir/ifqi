import ifqi.envs as envs
from ifqi.fqi.FQI import FQI
from ifqi.evaluation import evaluation
import ifqi.evaluation.evaluation as evaluate
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesRegressor
import matplotlib.pyplot as plt
import numpy as np

mdp = envs.Acrobot()
state_dim, action_dim = envs.get_space_info(mdp)
regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split': 5,
                    'min_samples_leaf': 2}
discrete_actions = mdp.action_space.values
regressor = ExtraTreesRegressor(**regressor_params)

# md.ActionRegressor(regressor, discreteActions=discrete_actions, decimals=5, **regressor_params)

#dataset = DatasetGenerator(mdp)
#dataset.generate(n_episodes=2000)

dataset = evaluation.collect_episodes(mdp, n_episodes=2000)
print('Dataset has %d samples' % dataset.shape[0])

reward_idx = state_dim + action_dim
sast = np.append(dataset[:, :reward_idx], dataset[:, reward_idx + 1:], axis=1)
r = dataset[:, reward_idx]

fqi = FQI(estimator=regressor,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=mdp.gamma,
          horizon=mdp.horizon,
          scaled=False,
          features=None,
          verbose=True)

fitParams = {}
# fitParams = {
#     "nEpochs": 300,
#     "batchSize": 50,
#     "validationSplit": 0.1,
#     "verbosity": False,
#     "criterion": "mse"
# }

initial_states = np.zeros((41, 4))
initial_states[:, 0] = np.linspace(-2, 2, 41)

fqi.partial_fit(sast, r, **fitParams)

iterations = 100
n_test_episodes = initial_states.shape[0]
iterationValues = []
for i in range(iterations - 1):
    fqi.partial_fit(None, None, **fitParams)

    values = evaluate.evaluate_policy(mdp, fqi, 1, initial_states=initial_states)
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
        plt.show()
