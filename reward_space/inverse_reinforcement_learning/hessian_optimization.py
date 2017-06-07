import cvxpy
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt

class HessianOptimizer(object):

    def __init__(self, hessians, threshold=0., features=None):
        self.hessians = hessians
        self.features = features
        self.threshold = threshold

        self.n_states_actions = self.hessians.shape[0]
        self.n_parameters = self.hessians.shape[1]

    def _build_weights_constraint(self, w, normalizer):
        if normalizer is not None:
            if normalizer == 'weighs_sum_to_one':
                norm_constraint = [cvxpy.sum_entries(w) == 1]
            elif normalizer == 'norm_weights_one':
                norm_constraint = [cvxpy.norm(w, 2) <= 1]
            elif normalizer == 'feature_max_one':
                norm_constraint = [cvxpy.max_entries(np.dot(self.features, w)) == 1,
                                   cvxpy.min_entries(np.dot(self.features, w)) == 0]
            else:
                norm_constraint = [normalizer]
        else:
            norm_constraint = []
        return norm_constraint

class TraceOptimizer(HessianOptimizer):

    def fit(self, normalizer=None):

        w = cvxpy.Variable(self.n_states_actions)
        final_hessian = self.hessians[0] * w[0]
        for i in range(1, self.n_states_actions):
            final_hessian += self.hessians[i] * w[i]

        objective = cvxpy.Minimize(cvxpy.trace(final_hessian))

        constraints = [final_hessian + self.threshold * np.eye(self.n_parameters) << 0]
        constraints.extend(self._build_weights_constraint(w, normalizer))

        problem = cvxpy.Problem(objective, constraints)
        result = problem.solve(solver='SCS', verbose=True)

        return w.value, final_hessian.value, result


class MaximumEigenvalueOptimizer(HessianOptimizer):

    def fit(self, normalizer=None):
        w = cvxpy.Variable(self.n_states_actions)
        final_hessian = self.hessians[0] * w[0]
        for i in range(1, self.n_states_actions):
            final_hessian += self.hessians[i] * w[i]

        #t = cvxpy.Variable()

        objective = cvxpy.Minimize(cvxpy.lambda_max(final_hessian))
        #objective = cvxpy.Minimize(t)

        constraints = []
        #constraints = [final_hessian - t * np.eye(self.n_parameters) << 0]
        constraints = [final_hessian + self.threshold * np.eye(self.n_parameters) << 0]
        constraints.extend(self._build_weights_constraint(w, normalizer))
        #print(constraints)

        problem = cvxpy.Problem(objective, constraints)
        result = problem.solve(solver='SCS', verbose=True)

        return w.value, final_hessian.value, result

class HeuristicOptimizerNegativeDefinite(HessianOptimizer):

    def fit(self, skip_check=False):
        traces = np.trace(self.hessians, axis1=1, axis2=2)

        if not skip_check:
            eigenvalues = la.eigh(self.hessians)[0]
            max_eigenvalues = eigenvalues[:, -1]
            if max_eigenvalues.max() > 1e-10:
                raise ValueError('Hessians must be negative semidefinite!')

        den = np.sqrt(np.sum(traces ** 2))
        w = - traces / den
        return w
