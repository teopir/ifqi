import numpy as np
import numpy.linalg as la
import scipy.optimize as opt

class DiscreteGaussian(object):

    '''
    @staticmethod
    def find_optimal_mu(x_opt, xs, sigma):
        inv_sigma = la.inv(sigma)
        fun = lambda mu: -DiscreteGaussian._pdf(x_opt, mu, sigma, inv_sigma, xs)
        grad = lambda mu: -DiscreteGaussian._grad(x_opt, mu, sigma, inv_sigma, xs)
        x0 = np.zeros(xs.shape[1])
        res = opt.minimize(fun, 3, jac=grad)
        return res
    '''
    @staticmethod
    def _myexp(x, mu, inv_sigma):
        return np.exp(-0.5 * la.multi_dot([(x - mu).T, inv_sigma, (x - mu)]))

    @staticmethod
    def _pdf(x, mu, sigma, inv_sigma, xs):
        num = DiscreteGaussian._myexp(x, mu, sigma)
        den = 0.
        for j in range(xs.shape[0]):
            den += DiscreteGaussian._myexp(xs[j, np.newaxis].T, mu, inv_sigma)
        return num / den

    def pdf(self, x):
        return DiscreteGaussian._pdf(x, self.mu, self.sigma, self.inv_sigma, self.xs)

    @staticmethod
    def _grad_log(x, mu, sigma, inv_sigma, xs):
        den = 0.
        num = 0.
        den = 0.
        for j in range(xs.shape[0]):
            val = DiscreteGaussian._myexp(x, mu, inv_sigma)
            num += (x - xs[j, np.newaxis].T) * val
            den += val
        return np.dot(inv_sigma, num / den)

    def __init__(self, xs, mu, sigma):
        self.xs = np.asarray(xs)
        self.mu = np.asarray(mu)
        self.sigma = np.asarray(sigma)
        self.inv_sigma = la.inv(self.sigma)

    def grad_log(self, x):
        return DiscreteGaussian._grad_log(x, self.mu, self.sigma, self.inv_sigma, self.xs)

    @staticmethod
    def _grad(x, mu, sigma, inv_sigma, xs):
        return DiscreteGaussian._pdf(x, mu, sigma, inv_sigma, xs) * DiscreteGaussian._grad_log(x, sigma, mu, inv_sigma, xs)

    def grad(self, x):
        return self.pdf(x) * self.grad_log(x)

'''
n = 6
v = np.arange(n-1).reshape((n-1,1))
xs = np.hstack([np.cos(v*2*np.pi/(n-1)), np.sin(v*2*np.pi/(n-1))])
xs = np.vstack([[0,0],xs])
mu = np.array([[0,0]], ndmin=2).T
sigma = 0.05*np.eye(2)
dg = DiscreteGaussian(xs, mu, sigma)
x = np.array([[0],[0]])

print(dg.pdf(x))
print(dg.grad_log(x))
'''