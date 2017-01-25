import numpy as np

from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_dataset, split_data_for_fqi
from ifqi.models.regressor import Regressor
from ifqi.algorithms.pbo.gradpbo import GradPBO

from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt

"""
Simple script to quickly run pbo. It solves the LQG environment.

"""

from optparse import OptionParser

# Input parameters by CLI
op = OptionParser(usage="usage: %prog input_file [options]",
                  version="%prog 1.0")
op.add_option("-f", action="store_false",
              dest="INCREMENTAL", default=True,
              help="Fixed version (non incremental).")
op.add_option("-T", default=1,
              dest="STEPS_HEAD", type="int",
              help="Step Heads")
op.add_option("-e", default=4,
              dest="EPOCH", type="int",
              help="Epochs")
op.add_option("-u", default=1,
              dest="UPDATE_EVERY", type="int",
              help="Update every.")
op.add_option("--indep", action="store_true",
              dest="INDEPENDENT", default=False,
              help="Independent.")
op.add_option("--activ", default="tanh",
              dest="ACTIVATION", type="str",
              help="NN activation")
(opts, args) = op.parse_args()

np.random.seed(6652)

mdp = envs.LQG1D()
mdp.seed(2897270658018522815)
state_dim, action_dim, reward_dim = envs.get_space_info(mdp)
reward_idx = state_dim + action_dim
discrete_actions = np.linspace(-8, 8, 20)
dataset = evaluation.collect_episodes(mdp, n_episodes=100)
check_dataset(dataset, state_dim, action_dim, reward_dim)

INCREMENTAL = opts.INCREMENTAL
ACTIVATION = opts.ACTIVATION
STEPS_AHEAD = opts.STEPS_HEAD
UPDATE_EVERY = opts.UPDATE_EVERY
INDEPENDENT = opts.INDEPENDENT
EPOCH = opts.EPOCH

print('INCREMENTAL:  {}'.format(INCREMENTAL))
print('ACTIVATION:   {}'.format(ACTIVATION))
print('STEPS_AHEAD:  {}'.format(STEPS_AHEAD))
print('UPDATE_EVERY: {}'.format(UPDATE_EVERY))
print('INDEPENDENT:  {}'.format(INDEPENDENT))

# sast, r = split_data_for_fqi(dataset, state_dim, action_dim, reward_dim)

### Q REGRESSOR ##########################
class LQG_Q(object):
    def model(self, s, a, omega):
        b = omega[:, 0]
        k = omega[:, 1]
        q = - b * b * s * a - 0.5 * k * a * a - 0.4 * k * s * s
        return q.ravel()

    def n_params(self):
        return 2

    def get_k(self, omega):
        b = omega[:, 0]
        k = omega[:, 1]
        return - b * b / k

    def name(self):
        return "R1"


q_regressor = LQG_Q()
##########################################

### F_RHO REGRESSOR ######################
n_q_regressors_weights = q_regressor.n_params()
Sequential.n_inputs = lambda self: n_q_regressors_weights


def _model_evaluation(self, theta):
    inv = theta
    for el in self.flattened_layers:
        # print(el)
        inv = el(inv)
    return inv


Sequential._model_evaluation = _model_evaluation
rho_regressor = Sequential()
rho_regressor.add(Dense(20, input_dim=n_q_regressors_weights, init='uniform', activation=ACTIVATION))
rho_regressor.add(Dense(n_q_regressors_weights, init='uniform', activation='linear'))
rho_regressor.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

import theano
import theano.tensor as T

theta = T.matrix()

res = rho_regressor._model_evaluation(theta)
ff = theano.function([theta], res)
h = np.array([[1, 2.]], dtype=theano.config.floatX)
assert np.allclose(ff(h), rho_regressor.predict(h))
# rho_regressor.fit(None, None)
##########################################

### PBO ##################################
pbo = GradPBO(bellman_model=rho_regressor,
              q_model=q_regressor,
              steps_ahead=STEPS_AHEAD,
              discrete_actions=discrete_actions,
              gamma=mdp.gamma,
              optimizer="adam",
              state_dim=state_dim,
              action_dim=action_dim,
              incremental=INCREMENTAL,
              update_theta_every=UPDATE_EVERY,
              verbose=1,
              independent=INDEPENDENT)


def tmetric(theta):
    t = pbo.apply_bop(theta[0], n_times=STEPS_AHEAD)
    return q_regressor.get_k(t)


state, actions, reward, next_states = split_dataset(dataset,
                                                    state_dim=state_dim,
                                                    action_dim=action_dim,
                                                    reward_dim=reward_dim)

theta0 = np.array([6., 10.001], dtype='float32').reshape(1, -1)
#theta0 = np.array([16., 10.001], dtype='float32').reshape(1, -1)
history = pbo.fit(state, actions, next_states, reward, theta0,
                  batch_size=10, nb_epoch=EPOCH,
                  theta_metrics={'k': tmetric})
##########################################
# Evaluate the final solution
initial_states = np.array([[1, 2, 5, 7, 10]]).T
values = evaluation.evaluate_policy(mdp, pbo, initial_states=initial_states)
print('Learned theta: {}'.format(pbo.learned_theta_value))
print('Final performance of PBO: {}'.format(values))

##########################################
# Some plot
ks = np.array(history.hist['k']).squeeze()
weights = np.array(history.hist['theta']).squeeze()

plt.figure()
plt.title('[train] evaluated weights')
plt.scatter(weights[:, 0], weights[:, 1], s=50, c=np.arange(weights.shape[0]),
            cmap='viridis', linewidth='0')
plt.xlabel('b')
plt.ylabel('k')
plt.colorbar()
plt.savefig(
    'LQG_MLP_{}_evaluated_weights_incremental_{}_activation_{}_steps_{}.png'.format(q_regressor.name(), INCREMENTAL,
                                                                                    ACTIVATION, STEPS_AHEAD),
    bbox_inches='tight')

plt.figure()
plt.plot(ks[30:-1])
plt.xlabel('iteration')
plt.ylabel('coefficient of max action (opt ~0.6)')
plt.savefig(
    'LQG_MLP_{}_max_coeff_incremental_{}_activation_{}_steps_{}.png'.format(q_regressor.name(), INCREMENTAL, ACTIVATION,
                                                                            STEPS_AHEAD),
    bbox_inches='tight')

theta = theta0.copy()
L = [np.array(theta)]
for i in range(STEPS_AHEAD * 200):
    theta = pbo.apply_bop(theta)
    L.append(np.array(theta))

L = np.array(L).squeeze()
print(L.shape)

print(theta)
print('K: {}'.format(q_regressor.get_k(theta)))
pbo.learned_theta_value = theta
values = evaluation.evaluate_policy(mdp, pbo, initial_states=initial_states)
print('Performance: {}'.format(values))

print('weights: {}'.format(rho_regressor.get_weights()))

plt.figure()
plt.scatter(L[:, 0], L[:, 1])
plt.title('Application of Bellman operator')
plt.xlabel('b')
plt.ylabel('k')
plt.savefig(
    'LQG_MLP_{}_bpo_application_incremental_{}_activation_{}_steps_{}.png'.format(q_regressor.name(), INCREMENTAL,
                                                                                  ACTIVATION, STEPS_AHEAD),
    bbox_inches='tight')

B_i, K_i = np.mgrid[-6:6:40j, -5:35:40j]
theta = np.column_stack((B_i.ravel(), K_i.ravel()))
theta_p = pbo.apply_bop(theta)

fig = plt.figure(figsize=(15, 10))
Q = plt.quiver(theta[:, 0], theta[:, 1], theta_p[:, 0] - theta[:, 0], theta_p[:, 1] - theta[:, 1], angles='xy')
plt.xlabel('b')
plt.ylabel('k')
plt.scatter(L[:, 0], L[:, 1], c='b')
plt.title('Gradient field - Act: {}, Inc: {}'.format(ACTIVATION, INCREMENTAL))
plt.savefig(
    'LQG_MLP_{}_grad_field_incremental_{}_activation_{}_steps_{}.png'.format(q_regressor.name(), INCREMENTAL,
                                                                             ACTIVATION, STEPS_AHEAD),
    bbox_inches='tight')
plt.show()
plt.close('all')




# best_rhos = pbo._rho_values[-1]
# q_w = np.array([4, 8])
# L = [q_w]
# for _ in range(10):
#     q_w = pbo._f2(best_rhos, q_w)
#     print(-q_w[1] ** 2 / q_w[0])
#     L.append(q_w)
# L = np.array(L)
# plt.figure()
# plt.scatter(L[:, 0], L[:, 1], s=50, c=np.arange(L.shape[0]), cmap='inferno')
#
# B_i, K_i = np.meshgrid(np.linspace(-10, 11, 20), np.linspace(-10, 11, 20))
# B_f = np.zeros(B_i.shape)
# K_f = np.zeros(K_i.shape)
# for i in range(B_i.shape[0]):
#     for j in range(K_i.shape[0]):
#         B_f[i, j], K_f[i, j] = pbo._f2(best_rhos, np.array([B_i[i, j], K_i[i, j]]))
#
# fig = plt.figure(figsize=(15, 10))
# Q = plt.quiver(B_i, K_i, B_f - B_i, K_f - K_i, angles='xy')
# plt.axis([-10, 10, -10, 10])
# plt.xlabel('b')
# plt.ylabel('k')
# plt.title('Theta vector field')
#
# plt.show()
