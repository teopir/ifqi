import numpy as np

from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset
from ifqi.models.regressor import Regressor
from ifqi.algorithms.pbo.PBO import PBO

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


class LQG_Q():
    def __init__(self, theta):
        self.theta = theta

    def predict(self, sa, **opt_pars):
        if 'f_rho' in opt_pars:
            k, b = opt_pars['f_rho']
        else:
            k, b = self.theta
        return - b * b * sa[:, 0] * sa[:, 1] - 0.5 * k * sa[:, 1] ** 2 - 0.4 * k * sa[:, 0] ** 2

theta = np.array([1., 0.])
regressor_params = {'theta': theta}
regressor = Regressor(LQG_Q, **regressor_params)

pbo = PBO(estimator=regressor,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=mdp.gamma,
          learning_steps=50,
          batch_size=10,
          learning_rate=1e-1,
          features={'name': 'None'},
          verbose=True)

thetas = pbo.fit(sast, r)
print('Best theta: ', thetas[-1])

initial_states = np.array([[1, 2, 5, 7, 10]]).T
values = evaluation.evaluate_policy(mdp, pbo, initial_states=initial_states)
print(values)
