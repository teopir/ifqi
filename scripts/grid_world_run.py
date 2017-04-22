import numpy as np
import numpy.linalg as la
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

import reward_space.utils.linalg2 as la2
from ifqi.envs.grid_world import GridWorld
from ifqi.evaluation import evaluation
from reward_space.policy_gradient.gradient_estimator import MaximumLikelihoodEstimator
from reward_space.utils.discrete_env_sample_estimator import DiscreteEnvSampleEstimator
from reward_space.utils.discrete_mdp_wrapper import DiscreteMdpWrapper
from scripts.policy import TabularPolicy, BoltzmannPolicy


def get_optimal_policy():
    pi = np.zeros((9, 4))
    pi[0, 2] = pi[1, 2] = pi[2, 1] \
        = pi[3, 1] = pi[4, 3] = pi[5, 1] \
        = pi[6, 2] = pi[7, 2] = pi[8, 0] = 1.
    return TabularPolicy(pi)

def build_state_features(mdp, binary=True):

    state_features = np.zeros((mdp.nS, 2))
    for i in range(mdp.nS):
        row, col = i / 3, i % 3
        state_features[i, :] = [row, col]

    if not binary:
        return state_features
    else:
        enc = OneHotEncoder(n_values=[3, 3], sparse=False,
                            handle_unknown='ignore')
        enc.fit(state_features)
        state_features_binary = enc.transform(state_features)
        return state_features_binary

def fit_maximum_likelihood_policy(state_features, optimal_action):
    '''
    Finds the maximum likelihood Boltzmann policy that best approximates a given
    deterministic policy.
    :param state_features: a (n_states, n_features) matrix representing the
                           feature vector for each state
    :param optimal_action: a (n_states,) vector containg for each state the index
                           of the optimal action
    :return: a pair: (action_weights, pi_prox)
             action_weights is a (n_actions, n_features) matrix representing the
                parameter estimated for each action
             pi_prox is a (n_states,n_actions) matrix representing the approximated
                probability distribution
    '''
    lr = LogisticRegression(penalty='l2',
                            tol=1e-20,
                            C=np.inf,
                            solver='newton-cg',
                            fit_intercept=False,
                            intercept_scaling=1,
                            max_iter=300,
                            multi_class='multinomial',
                            verbose=0,
                            n_jobs=1)

    lr.fit(state_features, optimal_action)
    action_weights = lr.coef_
    pi_prox = lr.predict_proba(state_features)
    return action_weights, pi_prox

if __name__ == '__main__':
    mdp = GridWorld()
    n_episodes = 100

    print('Building optimal policy...')
    opt_policy = get_optimal_policy()
    pi_opt = opt_policy.pi

    print('Building state features...')
    state_features = build_state_features(mdp, binary=True)

    print('Computing maximum likelihood Boltrzman policy...')
    action_weights, pi_prox = fit_maximum_likelihood_policy(state_features,
                                                            pi_opt.argmax(axis=1))

    d_kl = np.sum(pi_opt * np.log(pi_opt / pi_prox + 1e-24))
    n_parameters = action_weights.shape[0] * action_weights.shape[1]
    print('Number of features %s Number of parameters %s' % (
    state_features.shape[1], n_parameters))
    print('KL divergence = %s' % d_kl)

    policy = BoltzmannPolicy(state_features, action_weights)

    print('Collecting samples from optimal approx policy...')
    dataset = evaluation.collect_episodes(mdp, policy, n_episodes)
    n_samples = dataset.shape[0]
    print('Dataset made of %s samples' % n_samples)

    mdp_wrap = DiscreteMdpWrapper(mdp, episodic=True)
    pi_opt = opt_policy.get_distribution()
    pi = policy.get_pi()
    G = policy.gradient_log()

    # Optimal deterministic policy
    mdp_wrap.set_policy(pi_opt)
    J_opt = mdp_wrap.compute_J()

    # Optimal approx policy
    mdp_wrap.set_policy(pi)
    d_sa_mu = mdp_wrap.compute_d_sa_mu()
    D = np.diag(d_sa_mu)
    R = mdp_wrap.non_ep_R
    J_true = mdp_wrap.compute_J()
    Q_true = mdp_wrap.compute_Q_function()
    pi_tilde = np.repeat(pi, mdp_wrap.nA, axis=0)
    A_true = (np.eye(mdp_wrap.nA * mdp_wrap.nS) - pi_tilde).dot(Q_true)
    V_true = mdp_wrap.compute_V_function()[:mdp_wrap.nS]

    # ---------------------------------------------------------------------------
    # Sample estimations of return and gradient
    print('-' * 100)

    print('Estimating return and gradient...')
    estimator = DiscreteEnvSampleEstimator(dataset,
                                           mdp_wrap.gamma,
                                           mdp_wrap.state_space,
                                           mdp_wrap.action_space)

    d_s_mu_hat = estimator.get_d_s_mu()
    d_sa_mu_hat = np.dot(pi.T, d_s_mu_hat)
    D_hat = np.diag(d_sa_mu_hat)
    J_hat = estimator.get_J()

    print('Expected reward opt det policy J_opt = %g' % J_opt)
    print('True expected reward approx opt policy J_true = %g' % J_true)
    print('Estimated expected reward approx opt policy J_hat = %g' % J_hat)

    grad_J_hat = la.multi_dot([G.T, D_hat, Q_true])
    grad_J_true = la.multi_dot([G.T, D, Q_true])
    print('Dimension of the subspace %s/%s' % (
    la.matrix_rank(np.dot(G.T, D)), n_parameters))
    print(
    'True policy gradient (2-norm) DJ_true = %s' % la.norm(grad_J_true, 2))
    print(
    'Estimated policy gradient (2-norm) DJ_hat = %s' % la.norm(grad_J_hat, 2))

    #----------------------------------------------------------------------------
    print('-' * 100)
    print('Computing Q-function approx space...')
    X = np.dot(G.T, D_hat)
    phi = la2.nullspace(X)

    psi = phi
    '''
    psi = phi * la.norm(R)
    print('Computing reward function approx space...')
    Y = np.dot(np.eye(mdp_wrap.nA * mdp_wrap.nS) - pi_tilde, phi)
    psi = la2.range(Y)
    '''
    # ---------------------------------------------------------------------------
    # Hessian estimation
    print('Estimating hessians...')
    H = policy.hessian_log()
    ml_estimator = MaximumLikelihoodEstimator(dataset)

    a_norm = (A_true / la.norm(A_true)) [:, np.newaxis]
    r_norm = (R / la.norm(R)) [:, np.newaxis]

    hessian_r = ml_estimator.estimate_hessian(r_norm, G, H, True,
                                              mdp_wrap.state_space,
                                              mdp_wrap.action_space)[0]
    hessian_a = ml_estimator.estimate_hessian(a_norm, G, H, True,
                                              mdp_wrap.state_space,
                                              mdp_wrap.action_space)[0]
    hessian_hat = ml_estimator.estimate_hessian(psi, G, H, True,
                                              mdp_wrap.state_space,
                                              mdp_wrap.action_space)

    from reward_space.inverse_reinforcement_learning.hessian_optimization import HeuristicOptimizerNegativeDefinite
    he = HeuristicOptimizerNegativeDefinite(hessian_hat)
    w = he.fit()

    best = np.dot(psi, w)
    best = best / (max(best) - min(best)) * 11
    best = best*0.5 +

    from reward_space.policy_gradient.policy_gradient_learner import PolicyGradientLearner
    learner = PolicyGradientLearner(mdp, policy, lrate=0.3, verbose=1, max_iter_opt=400, tol_eval=0., tol_opt=-1.)
    '''
    theta0 = np.zeros((n_parameters, 1))
    policy.set_parameter(theta0)
    theta = learner.optimize(theta0)
    policy.set_parameter(theta)
    '''

    theta0 = np.zeros((n_parameters, 1))
    policy.set_parameter(theta0)
    theta = learner.optimize(theta0, reward=lambda traj: best[map(int,traj[:, 0]*4 + traj[:, 1])])
    policy.set_parameter(theta)

    '''

    from reward_space.inverse_reinforcement_learning.hessian_optimization import TraceOptimizer, MaximumEigenvalueOptimizer
    optimizer = MaximumEigenvalueOptimizer(hessian_hat, threshold=-1., features=psi)
    res = optimizer.fit()
    '''