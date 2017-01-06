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

    def get_k(self, omega):
        b = omega[:, 0]
        k = omega[:, 1]
        return - b * b / k

    def set_weights(self, w):
        self.w = np.array(w)

    def count_params(self):
        return self.w.size

q_regressor_params = dict()
q_regressor = Regressor(LQG_Q, **q_regressor_params)

ACTIVATION = 'sigmoid'
INCREMENTAL = False
### F_RHO REGRESSOR ######################
n_q_regressors_weights = q_regressor._regressor.count_params()
rho_regressor_params = {'n_input': n_q_regressors_weights,
                        'n_output': n_q_regressors_weights,
                        'hidden_neurons': [20],
                        'init': 'uniform',
                        'loss': 'mse',
                        'activation': ACTIVATION,
                        'optimizer': 'rmsprop',
                        'metrics': ['accuracy'],
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
          incremental=INCREMENTAL,
          verbose=True)

weights = pbo.fit(sast, r)
##########################################

initial_states = np.array([[1, 2, 5, 7, 10]]).T
values = evaluation.evaluate_policy(mdp, pbo, initial_states=initial_states)

print(values)

from matplotlib import pyplot as plt
weights = np.array(weights)
plt.subplot(1, 3, 1)
plt.title('[train] evaluated weights')
plt.xlabel('b')
plt.ylabel('k')
plt.scatter(weights[:, 1], weights[:, 0], s=50, c=np.arange(weights.shape[0]), cmap='inferno')
plt.colorbar()

best_rhos = pbo._rho_values[-1]
ks = q_regressor._regressor.get_k(np.array(pbo._q_weights_list))
plt.subplot(1, 3, 2)
plt.xlabel('iteration')
plt.ylabel('coefficient of max action (opt ~0.6)')
plt.plot(ks)
plt.grid()

B_i, K_i = np.meshgrid(np.linspace(-10, 11, 20), np.linspace(-10, 11, 20))
B_f = np.zeros(B_i.shape)
K_f = np.zeros(K_i.shape)
for i in range(B_i.shape[0]):
    for j in range(K_i.shape[0]):
        K_f[i, j], B_f[i, j] = pbo._f2(best_rhos, np.array([K_i[i, j], B_i[i, j]]))

plt.subplot(1, 3, 3)
plt.quiver(B_i, K_i, B_f - B_i, K_f - K_i, angles='xy')
plt.axis([-10, 10, -10, 10])
plt.xlabel('b')
plt.ylabel('k')
plt.title('Theta vector field')

plt.show()