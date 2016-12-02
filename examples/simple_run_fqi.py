import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor

from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_data_for_fqi
from ifqi.algorithms.fqi import FQI
from ifqi.models.actionregressor import ActionRegressor
from ifqi.models.regressor import Regressor
from ifqi.models.mlp import MLP
from ifqi.models.ensemble import Ensemble

"""
Simple script to quickly run fqi. It solves the Acrobot environment according
to the experiment presented in:

Ernst, Damien, Pierre Geurts, and Louis Wehenkel.
"Tree-based batch mode reinforcement learning."
Journal of Machine Learning Research 6.Apr (2005): 503-556.
"""

mdp = envs.Acrobot()
state_dim, action_dim, reward_dim = envs.get_space_info(mdp)
assert reward_dim == 1
regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'input_scaled': False,
                    'output_scaled': False}
discrete_actions = mdp.action_space.values

# ExtraTrees
regressor = Regressor(ExtraTreesRegressor, **regressor_params)

# Action regressor of Ensemble of ExtraTreesEnsemble
# regressor = Ensemble(ExtraTreesRegressor, **regressor_params)
# regressor = ActionRegressor(regressor, discrete_actions=discrete_actions,
#                            decimals=5, **regressor_params)

dataset = evaluation.collect_episodes(mdp, n_episodes=20)
check_dataset(dataset, state_dim, action_dim, reward_dim) # this is just a
# check, it can be removed in experiments
print('Dataset has %d samples' % dataset.shape[0])

# reward_idx = state_dim + action_dim
# sast = np.append(dataset[:, :reward_idx],
#                  dataset[:, reward_idx + reward_dim:-1],
#                  axis=1)
# r = dataset[:, reward_idx]
sast, r = split_data_for_fqi(dataset, state_dim, action_dim, reward_dim)

fqi_iterations = mdp.horizon  # this is usually less than the horizon
fqi = FQI(estimator=regressor,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=mdp.gamma,
          horizon=fqi_iterations,
          features=None,
          verbose=True)

fit_params = {}
# fit_params = {
#     "n_epochs": 300,
#     "batch_size": 50,
#     "validation_split": 0.1,
#     "verbosity": False,
#     "criterion": "mse"
# }

initial_states = np.zeros((41, 4))
initial_states[:, 0] = np.linspace(-2, 2, 41)

fqi.partial_fit(sast, r, **fit_params)

iterations = 100
iteration_values = []
for i in range(iterations - 1):
    fqi.partial_fit(None, None, **fit_params)
    values = evaluation.evaluate_policy(mdp, fqi, initial_states=initial_states)
    print(values)
    iteration_values.append(values[0])

    if i == 1:
        fig1 = plt.figure(1)
        ax = fig1.add_subplot(1, 1, 1)
        h = ax.plot(range(i + 1), iteration_values, 'ro-')
        plt.ylim(min(iteration_values), max(iteration_values))
        plt.xlim(0, i + 1)
        plt.ion()  # turns on interactive mode
        plt.show()
    elif i > 1:
        h[0].set_data(range(i + 1), iteration_values)
        ax.figure.canvas.draw()
        plt.ylim(min(iteration_values), max(iteration_values))
        plt.xlim(0, i + 1)
        plt.show()
