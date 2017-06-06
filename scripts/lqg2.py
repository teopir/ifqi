from ifqi.envs.lqg import LQG
from policy import GaussianPolicy
from ifqi.evaluation import evaluation
from scipy.linalg import solve_discrete_are
import numpy as np
import numpy.linalg as la

lqg = LQG(dimensions=2)

#Compute the optimal parameter
A, B, Q, R = lqg.A, lqg.B, lqg.Q, lqg.R
X = solve_discrete_are(A, B, Q, R)
K = -la.multi_dot([la.inv(la.multi_dot([B.T, X, B]) + R), B.T, X, A])

n_episodes_ex = 100
policy_ex = GaussianPolicy(K, covar=0.0 * np.eye(2))
trajectories_ex = evaluation.collect_episodes(lqg, policy_ex, n_episodes_ex)
print(np.dot(trajectories_ex[:, 4], trajectories_ex[:, 7])/n_episodes_ex)

k1 = np.dot(trajectories_ex[:, 0], trajectories_ex[:, 2]) / np.dot(trajectories_ex[:, 0], trajectories_ex[:, 0])
k2 = np.dot(trajectories_ex[:, 1], trajectories_ex[:, 3]) / np.dot(trajectories_ex[:, 1], trajectories_ex[:, 1])
n_episodes_ml = 100
policy_ml = GaussianPolicy(K = [[k1, 0], [0, k2]], covar=0.1 * np.eye(2))
trajectories_ml = evaluation.collect_episodes(lqg, policy_ml, n_episodes_ml)
print(np.dot(trajectories_ml[:, 4], trajectories_ml[:, 7])/n_episodes_ml)

G = policy_ml.gradient_log(trajectories_ex[:, :2], trajectories_ex[:, 2:4], type_='list')
#H = policy_ml.hessian_log(trajectories_ex[:, :2], trajectories_ex[:, 2:4])