import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_data_for_fqi
from ifqi.algorithms.generalizedfqi import GenGradFQI

np.random.seed(6652)

mdp = envs.LQG1D()
mdp.seed(2897270658018522815)
state_dim, action_dim, reward_dim = envs.get_space_info(mdp)
reward_idx = state_dim + action_dim
discrete_actions = np.linspace(-8, 8, 20)
dataset = evaluation.collect_episodes(mdp, n_episodes=100)
check_dataset(dataset, state_dim, action_dim, reward_dim)

### Q REGRESSOR ##########################
class LQG_Q(object):
    def __init__(self, init_theta):
        self.theta = theano.shared(value=np.array(init_theta, dtype=theano.config.floatX),
                                   borrow=True, name='theta')
        self.inputs = [T.dmatrix()]
        self.outputs = [self.model(self.inputs[0])]
        self.trainable_weights = [self.theta]

    def model(self, X):
        q = - (self.theta[0] ** 2) * X[:, 0] * X[:, 1] - 0.5 * (self.theta[1]) * X[:, 1] * X[:, 1]\
            - 0.4 * (self.theta[1]) * X[:, 0] * X[:, 0]
        return q

    def predict(self, X, **kwargs):
        if not hasattr(self, "eval_f"):
            self.eval_f = theano.function(self.inputs, self.outputs[0])
        return self.eval_f(X).ravel()

    def get_k(self, theta):
        if isinstance(theta, list):
            theta = theta[0].eval()
        b = theta[:, 0]
        k = theta[:, 1]
        return - b * b / k

    def get_weights(self):
        return self.theta.eval()

    def name(self):
        return "R1"

sast, r = split_data_for_fqi(dataset, state_dim, action_dim, reward_dim)

reg = LQG_Q([6., 10.001])
fqi = GenGradFQI(estimator=reg,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=mdp.gamma,
          update_theta_every=50,
          verbose=1)

def tmetric(theta):
    return reg.get_k(theta.reshape(-1,2))

fqi.fit(sast, r,
        batch_size=1, nb_epoch=3,
                            theta_metrics = {'k': tmetric}
)