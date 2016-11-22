import numpy as np

from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.algorithms.pbo.PBO import PBO
from pybrain.optimization import ExactNES

mdp = envs.LQG1D()
state_dim, action_dim, reward_dim = envs.get_space_info(mdp)
actions = np.array([-8, -7, -6, -5, -4, -3, -2.5, -2, -1.5, -1, -.75, -.5,
                    -.25, 0, .25, .5, .75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8])
dataset = evaluation.collect_episodes(mdp, n_episodes=200)

theta = np.array([1., 0.])
rho = np.array([1., 0.])

pbo = PBO(theta,
          state_dim,
          action_dim,
          actions,
          mdp.gamma,
          mdp.horizon)

epsilon = 1e-5
delta = np.inf
while delta > epsilon:
    optimizer = ExactNES(pbo.fitness, rho, minimize=True,
                         desiredEvaluation=1e-8)
    rho, score = optimizer.learn()

    old_theta = pbo.theta
    pbo.theta = pbo.f(rho)
    delta = np.sum(pbo.theta - old_theta) ** 2

    print(delta)

print(pbo.theta)
