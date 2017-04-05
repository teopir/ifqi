import numpy.linalg as la
from reward_space.policy_gradient.gradient_estimator2 import ReinforceGradientEstimator

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