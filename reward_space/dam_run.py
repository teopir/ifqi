from ifqi.envs.dam import Dam
from policy import GaussianPolicy1D, SimplePolicy
from ifqi.evaluation import evaluation
from utils.continuous_env_sample_estimator import ContinuousEnvSampleEstimator
from utils.utils import chebvalNd, MinMaxScaler, remove_projections, compute_feature_matrix, find_basis
import numpy as np
import utils.linalg2 as la2
import numpy.linalg as la

class GradientFeatureConstructor(object):

    def __init__(self, C):
        self.C = C

    def construct_reward_basis(self, d_sasa_mu, orthogonal=False):
        return la2.nullspace(np.dot(C.T, d_sasa_mu))

    def construct_Q_basis(self, d_sa_mu, orthogonal=False):
        return la2.nullspace(np.dot(C.T, np.diag(d_sa_mu)))


mdp = Dam(dreward=1, penalize=False)
opt_policy = SimplePolicy(K=1)
policy = GaussianPolicy1D(K=1, sigma=0.1)

n_episodes = 100
dataset_opt = evaluation.collect_episodes(mdp, opt_policy, n_episodes)
dataset = evaluation.collect_episodes(mdp, policy, n_episodes)

J_opt = 1.0 / n_episodes * np.sum(dataset_opt[:, 2] * dataset_opt[:, 4])
J_true = 1.0 / n_episodes * np.sum(dataset[:, 2] * dataset[:, 4])

estimator = ContinuousEnvSampleEstimator(dataset, mdp.gamma)
C = np.asarray(map(lambda s,a: policy.gradient_log_pdf(s,a), dataset[:, 0], dataset[:, 1]))
d_sa_mu = estimator.get_d_sa_mu()
d_sasa_mu = estimator.get_d_sasa_mu()

#fc = GradientFeatureConstructor(C)
#Phi_R = fc.construct_reward_basis(d_sasa_mu)

#R_hat, w, rmse, _ = la2.lsq(Phi_R, dataset[:, 2], w = d_sa_mu)

states = dataset[:, 0]
actions = dataset[:, 1]
rewards = dataset[:, 2]
discounts = dataset[:, 3]

bounds = [[0,states.max()], [0 ,actions.max()]]
scaler = MinMaxScaler(ndim=2, input_ranges=bounds)
scaled_states_actions = scaler.scale(dataset[:, :2])
scaled_states = scaled_states_actions[:,0]
scaled_actions = scaled_states_actions[:,1]


#Compute feature matrix

complement = [lambda x : policy.gradient_log_pdf(x[0],x[1])]
max_degree = 10
degrees = [[ds,da] for ds in range(max_degree+1) for da in range(max_degree+1)]
cheb_basis = map(lambda d: lambda x: chebvalNd(x, d), degrees)
n_samples = dataset.shape[0]
n_features = len(cheb_basis)
n_complement = 1

X = compute_feature_matrix(n_samples, n_features, scaled_states, scaled_actions, cheb_basis)
C = compute_feature_matrix(n_samples, n_complement, states, actions, complement)

W = estimator.d_sasa_mu
X_ort = remove_projections(X, C, W)
#X_ort = remove_projections(X, C, np.diag(discounts))
#Non mi interessa che sia ortonormale!!!
#X_ort_ort = find_basis(X_ort, np.diag(discounts))
#print('Rank of feature matrix X %s/%s' % (X_ort_ort.shape[1], X.shape[1]))

rewards_hat, w, rmse, _ = la2.lsq(X_ort, rewards)
error = np.abs(rewards - rewards_hat)
mae = np.mean(error)
error_rel = np.abs((rewards - rewards_hat)/rewards)
mare = np.mean(error_rel)

grad_J_true = 1.0/n_episodes * la.multi_dot([C.T, W, rewards])
grad_J_hat = 1.0/n_episodes * la.multi_dot([C.T, W, rewards_hat])
J_hat = 1.0/n_episodes * np.sum(rewards * discounts)
print('Results of LS rmse = %s mae = %s mare = %s' % (rmse, mae, mare))
print('True policy gradient %s' % grad_J_true)
print('Estimated policy gradient %s' % grad_J_hat)
print('Estimated expected return %s' % J_hat)
