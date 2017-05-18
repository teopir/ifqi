import numpy as np
import numpy.linalg as la
from sklearn.neighbors import NearestNeighbors

class ProtoValueFunctionsEstimator(object):

    '''
    Abstract class for proto value functions estimator
    '''

    eps = 1e-24

    def fit(self, dataset):
        pass

    def transform(self, k):
        pass

    def get_operator(self):
        return self.L

    def get_operator_type(self):
        return self.operator

    def get_adjacency_matrix(self):
        return self.W


class DiscreteProtoValueFunctionsEstimator(ProtoValueFunctionsEstimator):

    '''
    This class implements an estimator for the proto value functions in the
    discrete domain. The available graph operators are:
    - random walk
    - combinatorial laplacian
    - normalized laplacian
    '''

    def __init__(self,
                 state_space,
                 action_space=None,
                 operator='norm-laplacian',
                 method='on-policy',
                 type_='state-action'):
        self.state_space = state_space
        self.action_space = action_space
        self.operator = operator
        self.method = method
        self.type_ = type_
        self.n_states = len(state_space)
        if action_space is not None:
            self.n_actions = len(action_space)
            self.n_states_actions = self.n_states * self.n_actions


    def fit(self, dataset, count_method='unique-undiscounted'):
        if self.type_ == 'state':
            self._fit_state_pvfs(dataset, count_method)
        elif self.type_ == 'state-action':
            self._fit_state_action_pvfs(dataset, count_method)
        else:
            raise NotImplementedError

    def _fit_state_pvfs(self, dataset, count_method):
        states = dataset[:, 0]
        discounts = dataset[:, 4]
        n_samples = dataset.shape[0]

        W = np.zeros((self.n_states, self.n_states))

        i = 0
        while i < n_samples:
            if dataset[i, -1] == 0:
                s_i = np.argwhere(self.state_space == states[i])
                s_next_i = np.argwhere(self.state_space == states[i + 1])

                idx, idx_next = s_i , s_next_i

                if count_method == 'unique-undiscounted':
                    if W[idx, idx_next] == 0:
                        if self.method == 'on-policy':
                            W[idx, idx_next] = 1
                        elif self.method == 'off-policy':
                            raise ValueError()
                elif count_method == 'count-undiscounted':
                    if self.method == 'on-policy':
                        W[idx, idx_next] = 1
                    elif self.method == 'off-policy':
                        raise ValueError()
                elif count_method == 'discounted':
                    if W[idx, idx_next] == 0:
                        if self.method == 'on-policy':
                            W[idx, idx_next] = discounts[i]
                        elif self.method == 'off-policy':
                            raise ValueError()
                else:
                    raise NotImplementedError

            i = i + 1

        # Adjcency matrix symmetrization
        W = .5 * (W + W.T)
        self.W = W

        d = W.sum(axis=1)
        D = np.diag(d)
        D1 = np.diag(np.power(d + self.eps, -0.5))

        # Compute the operator
        if self.operator == 'norm-laplacian':
            self.L = np.eye(self.n_states) - la.multi_dot([D1, W, D1])
        elif self.operator == 'comb-laplacian':
            self.L = D - W
        elif self.operator == 'random-walk':
            self.L = la.solve(np.diag(d + self.eps), W)
        else:
            raise NotImplementedError

        # Diagonalize the operator
        if np.allclose(self.L.T, self.L):
            eigval, eigvec = la.eigh(self.L)
        else:
            eigval, eigvec = la.eig(self.L)
            eigval, eigvec = abs(eigval), abs(eigvec)
            ind = eigval.argsort()[::-1]
            eigval, eigvec = eigval[ind], eigvec[ind]

        self.eigval = eigval
        self.eigvec = eigvec

    def _fit_state_action_pvfs(self, dataset, count_method):
        states = dataset[:, 0]
        actions = dataset[:, 1]
        discounts = dataset[:, 4]
        n_samples = dataset.shape[0]

        W = np.zeros((self.n_states_actions, self.n_states_actions))

        i = 0
        while i < n_samples:
            if dataset[i, -1] == 0:
                s_i = np.argwhere(self.state_space == states[i])
                a_i = np.argwhere(self.action_space == actions[i])
                s_next_i = np.argwhere(self.state_space == states[i + 1])
                a_next_i = np.argwhere(self.action_space == actions[i + 1])

                idx, idx_next = s_i * self.n_actions + a_i, s_next_i * self.n_actions + a_next_i

                if count_method == 'unique-undiscounted':
                    if W[idx, idx_next] == 0:
                        if self.method == 'on-policy':
                            W[idx, idx_next] = 1
                        elif self.method == 'off-policy':
                            W[idx, s_next_i * self.n_actions + self.action_space] = 1
                elif count_method == 'count-undiscounted':
                    if self.method == 'on-policy':
                        W[idx, idx_next] = 1
                    elif self.method == 'off-policy':
                        W[idx, s_next_i * self.n_states + self.action_space] = 1
                elif count_method == 'discounted':
                    if W[idx, idx_next] == 0:
                        if self.method == 'on-policy':
                            W[idx, idx_next] = discounts[i]
                        elif self.method == 'off-policy':
                            W[idx, s_next_i * self.n_states + self.action_space] = discounts[i]
                else:
                    raise NotImplementedError

            i = i + 1

        #Adjcency matrix symmetrization
        W = .5 * (W + W.T)
        self.W = W

        d = W.sum(axis=1)
        D = np.diag(d)
        D1 = np.diag(np.power(d + self.eps, -0.5))

        #Compute the operator
        if self.operator == 'norm-laplacian':
            self.L = np.eye(self.n_states_actions) - la.multi_dot([D1, W, D1])
        elif self.operator == 'comb-laplacian':
            self.L = D - W
        elif self.operator == 'random-walk':
            self.L = la.solve(np.diag(d + self.eps), W)
        else:
            raise NotImplementedError

        #Diagonalize the operator
        if np.allclose(self.L.T, self.L):
            eigval, eigvec = la.eigh(self.L)
        else:
            eigval, eigvec = la.eig(self.L)
            eigval, eigvec = abs(eigval), abs(eigvec)
            ind = eigval.argsort()[::-1]
            eigval, eigvec = eigval[ind], eigvec[ind]

        self.eigval = eigval
        self.eigvec = eigvec

    def transform(self, k=None):
        if k is None:
            return self.eigval, self.eigvec
        else:
            return self.eigval[:k], self.eigvec[:, :k]

    def get_operator(self):
        return self.L

    def get_operator_type(self):
        return self.operator

    def get_adjacency_matrix(self):
        return self.W

class ContinuousProtoValueFunctions(ProtoValueFunctionsEstimator):

    def __init__(self,
                 operator='norm-laplacian',
                 method='on-policy',
                 type_='state-action',
                 k_neighbors=5,
                 kernel=None):
        self.operator = operator
        self.method = method
        self.type_ = type_
        self.k_neighbors= k_neighbors
        self.kernel = kernel

    def fit(self, dataset):

        self.knn = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.knn.fit(dataset[:, :2])
        distance_matrix = self.knn.kneighbors_graph(dataset[:, :2], mode='distance')
        similarity_matrix = distance_matrix.copy()
        similarity_matrix.data = self.kernel(similarity_matrix.data)
        W = similarity_matrix.toarray()
        W = .5 * (W + W.T)
        self.W = W

        d = W.sum(axis=1)
        D = np.diag(d)
        D1 = np.diag(np.power(d + self.eps, -0.5))

        # Compute the operator
        if self.operator == 'norm-laplacian':
            self.L = np.eye(W.shape[0]) - la.multi_dot([D1, W, D1])
        elif self.operator == 'comb-laplacian':
            self.L = D - W
        elif self.operator == 'random-walk':
            self.L = la.solve(np.diag(d + self.eps), W)
        else:
            raise NotImplementedError

        # Diagonalize the operator
        if np.allclose(self.L.T, self.L):
            eigval, eigvec = la.eigh(self.L)
        else:
            eigval, eigvec = la.eig(self.L)
            eigval, eigvec = abs(eigval), abs(eigvec)
            ind = eigval.argsort()[::-1]
            eigval, eigvec = eigval[ind], eigvec[ind]

        self.eigval = eigval
        self.eigvec = eigvec


    def transform(self, k=None):
        if k is None:
            return self.eigval, self.eigvec
        else:
            return self.eigval[:k], self.eigvec[:, :k]

