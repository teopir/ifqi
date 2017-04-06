import numpy.linalg as la
import numpy as np
from ifqi.evaluation import evaluation

class PolicyGradientLearner(object):

    def __init__(self,
                 mdp,
                 policy,
                 lrate=0.1,
                 lrate_decay=0,
                 estimator='reinforce',
                 max_iter_eval=100,
                 tol_eval=1e-3,
                 max_iter_opt=100,
                 tol_opt=1e-3):
        self.mdp = mdp
        self.policy = policy
        self.lrate = lrate
        self.lrate_decay = lrate_decay
        self.max_iter_eval = max_iter_eval
        self.tol_eval = tol_eval
        self.max_iter_opt = max_iter_opt
        self.tol_opt = tol_opt

        if estimator == 'reinforce':
            self.estimator = ReinforceGradientEstimator(self.mdp, self.policy,
                                                self.tol_eval,
                                                self.max_iter_eval)
        else:
            raise NotImplementedError

    def optimize(self, theta0):
        ite = 0
        theta = theta0

        print('Policy gradient: starting optimization...')

        gradient = self.estimator.estimate(theta0)
        gradient_norm = la.norm(gradient)

        print('Ite %s: gradient norm %s' % (ite, gradient_norm))
        while gradient_norm > self.tol_opt and ite < self.max_iter_opt:
            theta += self.lrate * gradient
            gradient = self.estimator.estimate(theta0)
            gradient_norm = la.norm(gradient)
            ite += 1
            print('Ite %s: gradient norm %s' % (ite, gradient_norm))

        return theta

class ReinforceGradientEstimator(object):

    def __init_(self,
                mdp,
                policy,
                tol=1e-3,
                max_iter=100):
        self.mdp = mdp
        self.policy = policy
        self.tol = tol
        self.max_iter = max_iter

    def estimate(self,theta):
        print('\tReinforce: starting estimation...')

        theta = np.array(theta, ndmin=1)
        dim = len(theta)
        self.policy.set_parameter(theta)

        ite = 0
        gradient_increment = np.inf
        gradient_estimate = np.inf

        print('\tIte %s gradient_increment %s' % (ite, gradient_increment))

        traj_return = []
        traj_log_grad = []

        while ite < self.max_iter and gradient_increment > self.tol:
            ite += 1

            traj = evaluation.collect_episodes(self.mdp, self.policy, 1)
            traj_return.append(np.sum(traj[:, 2] * traj[:, 4]))
            traj_log_grad.append(self.policy.log_grad(traj[:, 0], traj[:, 1]))

            baseline = np.sum(traj_log_grad ** 2 * traj_return) / np.sum(
                traj_log_grad ** 2)
            old_gradient_estimate = gradient_estimate
            gradient_estimate = 1. / ite * np.sum(
                traj_log_grad * (traj_return - baseline))

            gradient_increment = la.norm(
                gradient_estimate - old_gradient_estimate)

            print('\tIte %s gradient_increment %s' % (ite, gradient_increment))

        return gradient_estimate, gradient_increment, ite