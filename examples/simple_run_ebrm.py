import numpy as np

from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_dataset, split_data_for_fqi
from ifqi.models.regressor import Regressor
from ifqi.algorithms.pbo.ebrm import EmpiricalBellmanResidualMinimization

from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from keras import activations

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
discrete_actions = np.linspace(-8, 8, 20).astype(theano.config.floatX)
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
            T_s = T.fmatrix()
            T_a = T.fmatrix()
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


# import keras
# from keras.models import Model
# from keras.layers import Input, Dense
# keras.backend.set_floatx(theano.config.floatX)
# print(theano.config.floatX)

# class LQG_Q_NN(object):
#     def __init__(self, input_dim,
#                  layers=[{'output_dim': 20, 'activation':'tanh'}, {'output_dim':1, 'activation':'linear'}]):
#         self.T_s = Input(shape=(1,))
#         self.T_a = Input(shape=(1,))
#         self.layer_opts = layers
#         self.layers = list()
#         self.trainable_weights = []
#         input_shape = (None, input_dim)
#         for el in layers:
#             l = Dense(**el)
#             l.build(input_shape)
#             super(Dense, l).build(input_shape)
#             self.layers.append(l)
#             input_shape = (None, el['output_dim'])
#         print(self.layers)
#         for el in self.layers:
#             self.trainable_weights += el.trainable_weights
#         print(self.trainable_weights)
#
#     def model(self, s, a):
#         s = s.reshape((-1,1), ndim=2)
#         a = a.reshape((-1,1), ndim=2)
#         inv = T.concatenate((s,a),axis=1)
#         for el in self.layers:
#             inv = el(inv)
#         return inv.ravel()
#
#     def evaluate(self, s, a):
#         if not hasattr(self, "eval_f"):
#             T_s = T.fmatrix()
#             T_a = T.fmatrix()
#             self.eval_f = theano.function([T_s, T_a], self.model(T_s, T_a))
#         return self.eval_f(s, a)
#
#     def get_k(self, theta):
#         return 1
#
#     def name(self):
#         return "R2"


class LQG_NN(object):
    def __init__(self, input_dim, output_dim,
                 layers=[20], activations=['relu']):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = layers
        self.activations = activations
        self.trainable_weights = self.init()

    def init(self):
        # define the shared parameters
        self.W, self.b = [], []
        params = []
        for i in range(len(self.layers)):
            self.W.append(self.init_weights((self.input_dim if i == 0 else self.layers[i - 1], self.layers[i]),
                                            sigma=0, name='W_{}'.format(i)))
            params.append(self.W[-1])
            self.b.append(theano.shared(value=np.zeros((self.layers[i],), dtype=theano.config.floatX),
                                        borrow=True, name='b_{}'.format(i)))
            params.append(self.b[-1])
        last_layer_dim = self.input_dim
        if len(self.layers) > 0:
            last_layer_dim = self.layers[-1]
        self.Wy = self.init_weights((last_layer_dim, self.output_dim), sigma=0, name='Wy')
        self.by = theano.shared(value=np.zeros((self.output_dim,), dtype=theano.config.floatX),
                                borrow=True, name='by')
        # params = self.W + self.b + [self.Wy, self.by]
        params += [self.Wy, self.by]
        return params

    def model(self, s, a):
        s = s.reshape((-1, 1), ndim=2)
        a = a.reshape((-1, 1), ndim=2)
        y = T.concatenate((s, a), axis=1)
        for i in range(len(self.layers)):
            act = activations.get(self.activations[i])
            y = act(T.dot(y, self.W[i]) + self.b[i])
        act = activations.get("linear")
        y = act(T.dot(y, self.Wy) + self.by)
        return y

    def floatX(self, arr):
        return np.asarray(arr, dtype=theano.config.floatX)

    def init_weights(self, shape, sigma=0.01, name=''):
        if sigma == 0:
            W_bound = np.sqrt(6. / (shape[0] + shape[1]))
            return theano.shared(self.floatX(np.random.uniform(low=-W_bound, high=W_bound, size=shape)),
                                 borrow=True, name=name)
        return theano.shared(self.floatX(np.random.randn(*shape) * sigma), borrow=True, name=name)

    def get_k(self, theta):
        return 1.

    def evaluate(self, s, a):
        if not hasattr(self, "eval_f"):
            T_s = T.fmatrix()
            T_a = T.fmatrix()
            self.eval_f = theano.function([T_s, T_a], self.model(T_s, T_a))
        return self.eval_f(s, a)


theta0 = np.array([6., 10.001], dtype=theano.config.floatX).reshape(1, -1)
# q_regressor = LQG_NN(2,1,layers=[4,4], activations=['tanh', 'sigmoid'])
q_regressor = LQG_Q(theta0)
##########################################

### PBO ##################################
pfpo = EmpiricalBellmanResidualMinimization(q_model=q_regressor,
                                            discrete_actions=discrete_actions,
                                            gamma=mdp.gamma,
                                            optimizer="Nadam",
                                            state_dim=state_dim,
                                            action_dim=action_dim)
state, actions, reward, next_states = split_dataset(dataset,
                                                    state_dim=state_dim,
                                                    action_dim=action_dim,
                                                    reward_dim=reward_dim)
history = pfpo.fit(state.astype(theano.config.floatX), actions.astype(theano.config.floatX),
                   next_states.astype(theano.config.floatX), reward.astype(theano.config.floatX),
                   batch_size=1, nb_epoch=3,
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

states = discrete_states = np.linspace(-10, 10, 20)
actions = discrete_actions = np.linspace(-8, 8, 20)

initial_states = np.array([[1, 2, 5, 7, 10]]).T


def make_grid(x, y):
    m = np.meshgrid(x, y, copy=False, indexing='ij')
    return np.vstack(m).reshape(2, -1).T


SA = make_grid(states, actions).astype(theano.config.floatX)
S, A = SA[:, 0].reshape(-1,1), SA[:, 1].reshape(-1,1)

L = q_regressor.evaluate(S, A)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(S, A, L)
plt.show()

plt.figure()
plt.title('[train] evaluated weights - theta_0 ={}'.format(theta0))
plt.scatter(weights[:, 0], weights[:, 1], s=50, c=np.arange(weights.shape[0]),
            cmap='viridis', linewidth='0')
plt.xlabel('b')
plt.ylabel('k')
plt.colorbar()
plt.savefig(
    'PFPO_LQG_MLP_{}_evaluated_weights.png'.format(q_regressor.name()),
    bbox_inches='tight')

plt.figure()
plt.plot(ks[30:-1])
plt.title('Coefficient Optimal Action - theta_0 ={}'.format(theta0))
plt.xlabel('iteration')
plt.ylabel('coefficient of max action (opt ~0.6)')
plt.savefig(
    'PFPO_LQG_MLP_{}_max_coeff.png'.format(q_regressor.name()),
    bbox_inches='tight')

plt.show()
