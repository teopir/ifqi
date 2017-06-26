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
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
import copy

def plot_state_action_function(f, title, states=None, actions=None, _cmap='coolwarm'):
    if states is None:
        states = np.arange(-10, 10.5, .5)
    if actions is None:
        actions = np.arange(-8, 8.5, .5)
    states, actions = np.meshgrid(states, actions)
    z = f(states.ravel(), actions.ravel()).reshape(states.shape)

    arr = np.vstack([states.ravel()[:, np.newaxis], actions.ravel()[:, np.newaxis], z.ravel()[:, np.newaxis]])

    np.savetxt('data/csv/lqg_crirl_map', arr, delimiter=';')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(states, actions, z, rstride=1, cstride=1,
                    cmap=plt.get_cmap(_cmap),
                    linewidth=0.3, antialiased=True)
    ax.set_title(title)

def get_knn_function_for_plot(knn, rescale=False):
    if rescale:
        return lambda states, actions: knn.predict \
         (np.hstack([states[:, np.newaxis], actions[:, np.newaxis]]), rescale=False)
    else:
        return lambda states, actions: knn.predict \
            (np.hstack([states[:, np.newaxis], actions[:, np.newaxis]]))

def get_knn_function_for_prediction(knn, rescale=False):
    if rescale:
        return lambda traj: knn.predict(traj[:, :4], rescale=False)
    else:
        return lambda traj: knn.predict(traj[:, :4])

def train(learner, states_actions, gaussian_kernel, k, penalty, feature, knn_penalty=None, plot=False):
    knn = KNeighborsRegressor(n_neighbors=k, weights=gaussian_kernel)
    knn.fit(states_actions, feature.ravel())

    if plot:
        if penalty:
            function = lambda states, actions: .5 * \
                        get_knn_function_for_plot(knn)(states, actions) *+ \
                        .5 * 1. / 2. * get_knn_function_for_plot(knn_penalty, True)(states, actions)
        else:
            function = get_knn_function_for_plot(knn)
        plot_state_action_function(function, '%sknn %s' % (k, '- penalty' if penalty else ''))

    if penalty:
        function = lambda traj: .5 * get_knn_function_for_prediction(knn)(traj) + \
                             .5 * 1. / 2. * get_knn_function_for_prediction(knn_penalty, True)(traj)
    else:
        function = get_knn_function_for_prediction(knn)



    theta, history = learner.optimize(-np.ones((2,1))*0.2, reward=function, return_history=True)
    return history


global_grad = []
global_hess = []

lqg = LQG(dimensions=2)

#Compute the optimal parameter
A, B, Q, R = lqg.A, lqg.B, lqg.Q, lqg.R
X = solve_discrete_are(A, B, Q, R)
K = -la.multi_dot([la.inv(la.multi_dot([B.T, X, B]) + R), B.T, X, A])

K1=K2=0
e1 = []
e2 = []
j = []
for _ in range(20):

    n_episodes_ex = 20
    policy_ex = GaussianPolicy(K, covar=0.01 * np.eye(2))
    print(K)
    trajectories_ex = evaluation.collect_episodes(lqg, policy_ex, n_episodes_ex)
    n_samples = trajectories_ex.shape[0]
    J = np.dot(trajectories_ex[:, 4], trajectories_ex[:, 7])/n_episodes_ex
    print(J)
    j.append(j)
    continue

    k1 = np.dot(trajectories_ex[:, 0], trajectories_ex[:, 2]) / np.dot(trajectories_ex[:, 0], trajectories_ex[:, 0])
    k2 = np.dot(trajectories_ex[:, 1], trajectories_ex[:, 3]) / np.dot(trajectories_ex[:, 1], trajectories_ex[:, 1])

    n_episodes_ml = 20
    #policy_ml = GaussianPolicy(K = [[k1, 0], [0, k2]], covar=0.1 * np.eye(2))
    #policy_ml = BivariateGaussianPolicy([k1, k2], np.sqrt([0.2,0.2]), 0.)
    policy_ml = BivariateGaussianPolicy(K.diagonal(), np.sqrt([0.01, 0.01]), 0.)
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

    e1.extend(eigenvalues_hat[:, 0])
    e2.extend(eigenvalues_hat[:, 1])
    continue
    #plt.scatter(eigenvalues_hat[:, 0], eigenvalues_hat[:, 1])

    from reward_space.inverse_reinforcement_learning.hessian_optimization import MaximumEigenvalueOptimizer, TraceOptimizer, HeuristicOptimizerNegativeDefinite
    me_opt = MaximumEigenvalueOptimizer(hessian_hat/1000.)
    w_me = me_opt.fit('norm_weights_one')

    tr_opt = TraceOptimizer(hessian_hat/1000.)
    w_tr = tr_opt.fit('norm_weights_one')

    hessians_def = np.argwhere(eigenvalues_hat[:, 0] * eigenvalues_hat[:, 1] > 0).ravel()
    hessians_nd = hessian_hat[hessians_def]
    hessians_nd[eigenvalues_hat[hessians_def, 0] > 0] *= -1
    he_opt = HeuristicOptimizerNegativeDefinite(hessians_nd)
    w_he_nd = he_opt.fit()

    w_he = np.zeros(hessian_hat.shape[0])
    w_he[hessians_def] = w_he_nd
    w_he[eigenvalues_hat[:,0] > 0] *= -1

    eco_me = np.array(np.dot(psi, w_me[0]))
    eco_tr = np.array(np.dot(psi, w_tr[0]))
    eco_he = np.dot(psi, w_he[:, np.newaxis])

    names = ['Reward', 'ECO-ME', 'ECO-TR', 'ECO-HE']
    basis_functions = [trajectories_ex[:, 4]/ la.norm(trajectories_ex[:, 4]), eco_me.ravel(), eco_tr.ravel(), eco_he.ravel()]

    '''
    #Gradient and hessian estimation
    '''

    #Rescale rewards into to have difference between max and min equal to 1

    scaler = MinMaxScaler()

    scaled_basis_functions = []
    for bf in basis_functions:
        sbf = scaler.fit_transform(bf[:, np.newaxis]).ravel()
        scaled_basis_functions.append(sbf)

    scaled_basis_functions = basis_functions
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
    '''
    print('Saving results...')
    gradients_np = np.vstack(gradients)
    print(hessians)
    hessians_np = np.vstack(hessians)

    print(hessians_np)
    saveme1 = np.zeros(3, dtype=object)
    saveme1[0] = names
    saveme1[1] = gradients_np
    saveme1[2] = hessians_np

    global_grad.append(gradients)
    global_hess.append(hessians)

    '''
    #REINFORCE training
    '''
    print('-' * 100)
    print('Estimating d(s,a)...')

    count_sa_hat = np.ones(n_samples)
    count_sa_hat /= count_sa_hat.max()

    from reward_space.utils.k_neighbors_regressor_2 import KNeighborsRegressor2
    count_sa_knn = KNeighborsRegressor2(n_neighbors=5, weights=gaussian_kernel)
    count_sa_knn.fit(trajectories_ex[:, :4], count_sa_hat)
    # plot_state_action_function(get_knn_function_for_plot(count_sa_knn, True), 'd(s,a)')


    print('-' * 100)
    print('Training with REINFORCE using true reward and true a function')

    iterations = 300

    from reward_space.policy_gradient.policy_gradient_learner import \
        PolicyGradientLearner

    learner = PolicyGradientLearner(lqg, policy_ml, max_iter_opt=iterations,
                                    lrate=0.002,
                                    gradient_updater='adam', verbose=1,
                                    tol_opt=-1., state_dim=2, action_dim=2)

    _, history = learner.optimize(-np.ones((2, 1)) * 0.2, return_history=True)
    histories = [history]

    for i in range(1, len(scaled_basis_functions)):
        print(names[i])
        sbf = scaled_basis_functions[i]
        history = train(learner, trajectories_ex[:, :4], gaussian_kernel, 2,
                        True, sbf,
                        count_sa_knn, False)
        histories.append(history)

    labels = names

    histories = np.array(histories)
    t = PrettyTable()
    t.add_column('Basis function', labels)
    t.add_column('Final parameter',
                 [np.array(histories)[i][-1, 0] for i in range(4)])
    t.add_column('Final return',
                 [np.array(histories)[i][-1, 1] for i in range(4)])
    t.add_column('Final gradient',
                 [np.array(histories)[i][-1, 2] for i in range(4)])
    print(t)

    plot = False
    if plot:
        _range = np.arange(iterations + 1)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_ylabel('k1')
        ax.set_xlabel('iterations')
        ax.set_zlabel('k2')
        ax.plot([0, iterations + 1], [k1, k1], [k2, k2], color='k',
                label='Optimal parameter')
        for i in range(len(histories)):
            ks = np.stack(np.array(histories)[i][:, 0]).squeeze()
            ax.plot(_range, ks[:, 0].ravel(), ks[:, 1].ravel(),
                    marker=None, label=labels[i])
        ax.legend(loc='upper right')

        _range = np.arange(iterations + 1)
        fig, ax = plt.subplots()
        ax.set_xlabel('parameter')
        ax.set_ylabel('iterations')
        fig.suptitle('REINFORCE - Return ')

        for i in range(len(histories)):
            ax.plot(_range, np.array(histories)[i][:, 1].ravel(), marker=None,
                    label=labels[i])
        ax.legend(loc='upper right')

    saveme2 = np.zeros(2, dtype=object)
    saveme2[0] = labels
    saveme2[1] = histories

    import time

    mytime = time.time()
    np.save('data/lqg/lqg2_gradients_hessians_%s' % mytime, saveme1)
    np.save('data/lqg/lqg2_comparision_%s' % mytime,  saveme2)

    '''

print(np.mean(j))

'''
gradients = np.mean(np.stack(global_grad).squeeze(), axis=0)
hessians = np.mean(np.stack(global_hess).squeeze(), axis=0)

fig = plt.figure()
k = np.array([k1, k2])
x = y = np.arange(-3.0, 3.0, 0.05)
x += k1
y += k2
X, Y = np.meshgrid(x, y)
for i in range(len(basis_functions)):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')

    zs = np.array([J + np.dot(k-np.array([x,y]), gradients[i].squeeze()) + la.multi_dot([k-np.array([x,y]), hessians[i], np.transpose(k-np.array([x,y]))])  for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_zlim(-10**4.5, 100)
    ax.set_xlabel('k1')
    ax.set_ylabel('k2')
    ax.set_zlabel('almost J')
    ax.set_title(names[i])
'''