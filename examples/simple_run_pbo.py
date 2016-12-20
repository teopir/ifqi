import numpy as np

from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset
from ifqi.models.regressor import Regressor
from ifqi.models.mlp import MLP
from ifqi.models.linear import Ridge
from ifqi.algorithms.pbo.PBO import PBO

"""
Simple script to quickly run pbo. It solves the LQG environment.

"""

mdp = envs.LQG1D()
state_dim, action_dim, reward_dim = envs.get_space_info(mdp)
reward_idx = state_dim + action_dim
discrete_actions = np.linspace(-8, 8, 20)
dataset = evaluation.collect_episodes(mdp, n_episodes=100)
check_dataset(dataset, state_dim, action_dim, reward_dim)
sast = np.append(dataset[:, :reward_idx],
                 dataset[:, reward_idx + reward_dim:-1],
                 axis=1)
r = dataset[:, reward_idx]

### Q REGRESSOR ##########################
class LQG_Q():
    def __init__(self):
        self.w = np.array([1., 0.])

    def predict(self, sa):
        k, b = self.w
        return - b * b * sa[:, 0] * sa[:, 1] - 0.5 * k * sa[:, 1] ** 2 - 0.4 * k * sa[:, 0] ** 2

    def get_weights(self):
        return self.w

    def set_weights(self, w):
        self.w = np.array(w)

    def count_params(self):
        return self.w.size

#q_regressor_params = dict()
#q_regressor = Regressor(LQG_Q, **q_regressor_params)
#phi = dict(name='poly', params=dict(degree=3))
#q_regressor_params = dict(features=phi)
#q_regressor = Regressor(Ridge, **q_regressor_params)
#q_regressor.fit(sast[:, :state_dim + action_dim], r)  # necessary to init Ridge
##########################################

q_regressor_params = {'n_input': 2,
                      'n_output': 1,
                      'hidden_neurons': [20, 20],
                      'activation': 'sigmoid',
                      'optimizer': 'rmsprop',
                      'input_scaled': 1}
q_regressor = Regressor(MLP, **q_regressor_params)

### F_RHO REGRESSOR ######################
n_q_regressors_weights = q_regressor._regressor.count_params()
rho_regressor_params = {'n_input': n_q_regressors_weights,
                        'n_output': n_q_regressors_weights,
                        'hidden_neurons': [20],
                        'activation': 'sigmoid',
                        'optimizer': 'rmsprop',
                        'input_scaled': 1}
rho_regressor = Regressor(MLP, **rho_regressor_params)
##########################################

### PBO ##################################
pbo = PBO(estimator=q_regressor,
          estimator_rho=rho_regressor,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=mdp.gamma,
          learning_steps=50,
          batch_size=10,
          learning_rate=1e-1,
          verbose=True)

weights = pbo.fit(sast, r)
##########################################

'''
from matplotlib import pyplot as plt
weights = np.array(weights)
plt.scatter(weights[:, 0], weights[:, 1], s=50, c=np.arange(weights.shape[0]), cmap='inferno')
plt.show()
'''

initial_states = np.array([[1, 2, 5, 7, 10]]).T
values = evaluation.evaluate_policy(mdp, pbo, initial_states=initial_states)
print(values)
