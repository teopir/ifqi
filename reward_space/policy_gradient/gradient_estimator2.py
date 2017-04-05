import numpy as np
import numpy.linalg as la
from ifqi.evaluation import evaluation


class GradientEstimator(object):
    def __init_(self,
                mdp,
                policy,
                tol=1e-3,
                max_iter=100):
        self.mdp = mdp
        self.policy = policy
        self.tol = tol
        self.max_iter = max_iter

    def estimate(theta):
        pass


class ReinforceGradientEstimator(GradientEstimator):

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