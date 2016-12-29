import numpy as np

from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_data_for_fqi
from ifqi.models.regressor import Regressor
from ifqi.models.mlp import MLP
from ifqi.models.linear import Ridge
from ifqi.algorithms.pbo.pbo import PBO

"""
Simple script to quickly run pbo. It solves the LQG environment.

"""

mdp = envs.LQG1D()
state_dim, action_dim, reward_dim = envs.get_space_info(mdp)
reward_idx = state_dim + action_dim
discrete_actions = np.linspace(-8, 8, 20)
dataset = evaluation.collect_episodes(mdp, n_episodes=100)
check_dataset(dataset, state_dim, action_dim, reward_dim)
sast, r = split_data_for_fqi(dataset, state_dim, action_dim, reward_dim)

### Q REGRESSOR ##########################
class LQG_Q():
    def __init__(self):
        self.w = np.array([1., 0.])

    def predict(self, sa):
        k, b = self.w
        #print(k,b)
        return - b * b * sa[:, 0] * sa[:, 1] - 0.5 * k * sa[:, 1] ** 2 - 0.4 * k * sa[:, 0] ** 2

    def get_weights(self):
        return self.w

    def set_weights(self, w):
        self.w = np.array(w)

    def count_params(self):
        return self.w.size

q_regressor_params = dict()
q_regressor = Regressor(LQG_Q, **q_regressor_params)
#phi = dict(name='poly', params=dict(degree=3))
#q_regressor_params = dict(features=phi)
#q_regressor = Regressor(Ridge, **q_regressor_params)
#q_regressor.fit(sast[:, :state_dim + action_dim], r)  # necessary to init Ridge
##########################################

# q_regressor_params = {'n_input': 2,
#                       'n_output': 1,
#                       'hidden_neurons': [20, 20],
#                       'activation': 'sigmoid',
#                       'optimizer': 'rmsprop',
#                       'input_scaled': 1}
# q_regressor = Regressor(MLP, **q_regressor_params)

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

print(weights)

initial_states = np.array([[1, 2, 5, 7, 10]]).T
values = evaluation.evaluate_policy(mdp, pbo, initial_states=initial_states)
print(values)

from matplotlib import pyplot as plt
weights = np.array(weights)
plt.scatter(weights[:, 0], weights[:, 1], s=50, c=np.arange(weights.shape[0]), cmap='inferno')

best_rhos = pbo._rho_values[-1]
q_w = np.array([4, 8])
L = [q_w]
for _ in range(10):
    q_w = pbo._f2(best_rhos, q_w)
    print(-q_w[1] ** 2 / q_w[0])
    L.append(q_w)
L = np.array(L)
plt.figure()
plt.scatter(L[:, 0], L[:, 1], s=50, c=np.arange(L.shape[0]), cmap='inferno')

B_i, K_i = np.meshgrid(np.linspace(-10, 11, 20), np.linspace(-10, 11, 20))
B_f = np.zeros(B_i.shape)
K_f = np.zeros(K_i.shape)
for i in range(B_i.shape[0]):
    for j in range(K_i.shape[0]):
        B_f[i, j], K_f[i, j] = pbo._f2(best_rhos, np.array([B_i[i, j], K_i[i, j]]))

fig = plt.figure(figsize=(15, 10))
Q = plt.quiver(B_i, K_i, B_f - B_i, K_f - K_i, angles='xy')
plt.axis([-10, 10, -10, 10])
plt.xlabel('b')
plt.ylabel('k')
plt.title('Theta vector field')

plt.show()