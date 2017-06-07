from ifqi.envs.lqg import LQG
from policy import GaussianPolicy, BivariateGaussianPolicy
from ifqi.evaluation import evaluation
from scipy.linalg import solve_discrete_are
import numpy as np
import numpy.linalg as la
import reward_space.utils.linalg2 as la2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from prettytable import PrettyTable
from mpl_toolkits.mplot3d import Axes3D

lqg = LQG(dimensions=2)

#Compute the optimal parameter
A, B, Q, R = lqg.A, lqg.B, lqg.Q, lqg.R
X = solve_discrete_are(A, B, Q, R)
K = -la.multi_dot([la.inv(la.multi_dot([B.T, X, B]) + R), B.T, X, A])

n_episodes_ex = 20
policy_ex = GaussianPolicy(K, covar=0.0 * np.eye(2))
trajectories_ex = evaluation.collect_episodes(lqg, policy_ex, n_episodes_ex)
n_samples = trajectories_ex.shape[0]
print(np.dot(trajectories_ex[:, 4], trajectories_ex[:, 7])/n_episodes_ex)

k1 = np.dot(trajectories_ex[:, 0], trajectories_ex[:, 2]) / np.dot(trajectories_ex[:, 0], trajectories_ex[:, 0])
k2 = np.dot(trajectories_ex[:, 1], trajectories_ex[:, 3]) / np.dot(trajectories_ex[:, 1], trajectories_ex[:, 1])
n_episodes_ml = 20
#policy_ml = GaussianPolicy(K = [[k1, 0], [0, k2]], covar=0.1 * np.eye(2))
policy_ml = BivariateGaussianPolicy([k1, k2], [np.sqrt(0.1), np.sqrt(0.1)], 0.)
trajectories_ml = evaluation.collect_episodes(lqg, policy_ml, n_episodes_ml)
print(np.dot(trajectories_ml[:, 4], trajectories_ml[:, 7])/n_episodes_ml)

G = np.stack(policy_ml.gradient_log(trajectories_ex[:, :2], trajectories_ex[:, 2:4], type_='list'))
H = np.stack(policy_ml.hessian_log(trajectories_ex[:, :2], trajectories_ex[:, 2:4], type_='list'), axis=0)
D_hat = np.diag(trajectories_ex[:, 7])

print('-' * 100)
print('Computing Q-function approx space...')

X = np.dot(G.T, D_hat)
phi = la2.nullspace(X)

print('-' * 100)
print('Computing A-function approx space...')

sigma_kernel = 1.
def gaussian_kernel(x):
    return 1. / np.sqrt(2 * np.pi * sigma_kernel ** 2) * np.exp(- 1. / 2 * x ** 2 / (sigma_kernel ** 2))


from sklearn.neighbors import NearestNeighbors
knn_states = NearestNeighbors(n_neighbors=10)
knn_states.fit(trajectories_ex[:, :2])
pi_tilde = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    pi_tilde[i, i] = policy_ml.pdf(trajectories_ex[i, :2], trajectories_ex[i, 2:4])
    _, idx = knn_states.kneighbors([trajectories_ex[i, :2]])
    idx = idx.ravel()
    for j in idx:
        pi_tilde[i, j] = policy_ml.pdf(trajectories_ex[j, :2], trajectories_ex[j, 2:4])

pi_tilde /= pi_tilde.sum(axis=1)

Y = np.dot(np.eye(n_samples) - pi_tilde, phi)
psi = la2.range(Y)


from reward_space.policy_gradient.gradient_estimator import MaximumLikelihoodEstimator
ml_estimator = MaximumLikelihoodEstimator(trajectories_ex)

print('Computing gradients and hessians...')
gradient_hat = ml_estimator.estimate_gradient(psi, G, use_baseline=True)
hessian_hat = ml_estimator.estimate_hessian(psi, G, H, use_baseline=True)

eigenvalues_hat, _ = la.eigh(hessian_hat)
traces_hat = np.trace(hessian_hat, axis1=1, axis2=2)

plt.scatter(eigenvalues_hat[:, 0], eigenvalues_hat[:, 1])

from reward_space.inverse_reinforcement_learning.hessian_optimization import MaximumEigenvalueOptimizer, TraceOptimizer, HeuristicOptimizerNegativeDefinite
me_opt = MaximumEigenvalueOptimizer(hessian_hat/100.)
w_me = me_opt.fit('norm_weights_one')

tr_opt = TraceOptimizer(hessian_hat/100.)
w_tr = tr_opt.fit('norm_weights_one')

hessians_def = np.argwhere(eigenvalues_hat[:, 0] * eigenvalues_hat[:, 1] > 0).ravel()
hessians_nd = hessian_hat[hessians_def]
hessians_nd[eigenvalues_hat[hessians_def, 0] > 0] *= -1
he_opt = HeuristicOptimizerNegativeDefinite(hessians_nd)
w_he_nd = he_opt.fit()

w_he = np.zeros(hessian_hat.shape[0])
w_he[hessians_def] = w_he_nd
w_he[eigenvalues_hat[hessians_def, 0] > 0] *= -1

eco_me = np.array(np.dot(psi, w_me[0]))
eco_tr = np.array(np.dot(psi, w_tr[0]))
eco_he = np.dot(psi, w_he[:, np.newaxis])

names = ['Reward', 'ECO-ME', 'ECO-TR', 'ECO-HE']
basis_functions = [trajectories_ex[:, 4], eco_me.ravel(), eco_tr.ravel(), eco_he.ravel()]

'''
Gradient and hessian estimation
'''

#Rescale rewards into to have difference between max and min equal to 1
scaler = MinMaxScaler()

scaled_basis_functions = []
for bf in basis_functions:
    sbf = scaler.fit_transform(bf).ravel()
    scaled_basis_functions.append(sbf)

gradients, hessians, eigvals, trace = [], [], [], []
print('Estimating gradient and hessians...')
for sbf in scaled_basis_functions:
    gradients.append(ml_estimator.estimate_gradient(sbf, G, use_baseline=True))
    hess = ml_estimator.estimate_hessian(sbf, G, H, use_baseline=True)[0]
    hessians.append(hess)
    eigvals.append(la.eigh(hess)[0])
    trace.append(np.trace(hess))

t = PrettyTable()
t.add_column('Basis function', names)
t.add_column('Gradient', gradients)
t.add_column('Hessian', hessians)
t.add_column('Eigenvalues', eigvals)
t.add_column('Trace', trace)
print(t)

fig = plt.figure()
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
for i in range(len(basis_functions)):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    zs = np.array([np.dot(eigvals[i], [x**2, y**2])  for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_zlim(-10**5.5, 100)
    ax.set_xlabel('k1')
    ax.set_ylabel('k2')
    ax.set_zlabel('almost J')
    ax.set_title(names[i])