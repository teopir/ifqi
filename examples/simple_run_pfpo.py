import numpy as np

from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_dataset, split_data_for_fqi
from ifqi.models.regressor import Regressor
from ifqi.algorithms.pbo.pfpo import PFPO

from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt

import theano
import theano.tensor as T

"""
Simple script to quickly run pbo. It solves the LQG environment.

"""

np.random.seed(6652)

mdp = envs.LQG1D()
mdp.seed(2897270658018522815)
state_dim, action_dim, reward_dim = envs.get_space_info(mdp)
reward_idx = state_dim + action_dim
discrete_actions = np.linspace(-8, 8, 20)
dataset = evaluation.collect_episodes(mdp, n_episodes=100)
check_dataset(dataset, state_dim, action_dim, reward_dim)

INCREMENTAL = True
ACTIVATION = 'tanh'


# sast, r = split_data_for_fqi(dataset, state_dim, action_dim, reward_dim)

### Q REGRESSOR ##########################
class LQG_Q(object):
    def __init__(self, init_theta):
        self.theta = theano.shared(value=np.array(init_theta, dtype=theano.config.floatX),
                                   borrow=True, name='theta')
        self.trainable_weights = [self.theta]

    def model(self, s, a):
        q = - self.theta[:, 0] ** 2 * s * a - 0.5 * self.theta[:, 1] * a * a - 0.4 * self.theta[:, 1] * s * s
        return q.ravel()

    def evaluate(self, s, a):
        if not hasattr(self, "eval_f"):
            T_s = T.matrix()
            T_a = T.matrix()
            self.eval_f = theano.function([T_s, T_a], self.model(T_s, T_a))
        return self.eval_f(s, a)

    def n_params(self):
        return 2

    def get_k(self, theta):
        if isinstance(theta, list):
            theta = theta[0].eval()
        b = theta[:, 0]
        k = theta[:, 1]
        return - b * b / k

    def name(self):
        return "R1"


theta0 = np.array([6., 10.001], dtype='float32').reshape(1, -1)
q_regressor = LQG_Q(theta0)
##########################################

### PBO ##################################
pfpo = PFPO(q_model=q_regressor,
            discrete_actions=discrete_actions,
            gamma=mdp.gamma,
            optimizer="adam",
            state_dim=state_dim,
            action_dim=action_dim)
state, actions, reward, next_states = split_dataset(dataset,
                                                    state_dim=state_dim,
                                                    action_dim=action_dim,
                                                    reward_dim=reward_dim)
history = pfpo.fit(state, actions, next_states, reward,
                   batch_size=1, nb_epoch=2,
                   theta_metrics={'k': lambda theta: q_regressor.get_k(theta)})
##########################################
# Evaluate the final solution
initial_states = np.array([[1, 2, 5, 7, 10]]).T
values = evaluation.evaluate_policy(mdp, pfpo, initial_states=initial_states)
print(values)

##########################################
# Some plot
ks = np.array(history.hist['k']).squeeze()
weights = np.array(history.hist['theta']).squeeze()

plt.figure()
plt.title('[train] evaluated weights - theta_0 ={}'.format(theta0))
plt.scatter(weights[:, 0], weights[:, 1], s=50, c=np.arange(weights.shape[0]),
            cmap='viridis', linewidth='0')
plt.xlabel('b')
plt.ylabel('k')
plt.colorbar()
plt.savefig(
    'PFPO_LQG_MLP_{}_evaluated_weights_incremental_{}_activation_{}.png'.format(q_regressor.name(), INCREMENTAL, ACTIVATION),
    bbox_inches='tight')

plt.figure()
plt.plot(ks[30:-1])
plt.title('Coefficient Optimal Action - theta_0 ={}'.format(theta0))
plt.xlabel('iteration')
plt.ylabel('coefficient of max action (opt ~0.6)')
plt.savefig(
    'PFPO_LQG_MLP_{}_max_coeff_{}_activation_{}.png'.format(q_regressor.name(), INCREMENTAL, ACTIVATION),
    bbox_inches='tight')

plt.show()