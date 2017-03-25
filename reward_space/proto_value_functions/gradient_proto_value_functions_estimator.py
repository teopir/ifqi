import numpy as np
import numpy.linalg as la

class GradientProtoValueFunctionsEstimator(object):

    def __init__(self,
                 pvf_estimator,
                 gradient_space,
                 weights,
                 method='1'):
        self.pvf_estimator = pvf_estimator
        self.gradient_space = gradient_space
        self.weights = weights
        self.method = method


    def fit(self):
        if self.method == 'project-laplacian':
            pass
        elif self.method == 'remove-projections':
            pass


    def _fit_project_laplacian(self):
        #Controllare che abbia senso
        #COntrollare che siano effettivamente ortogonali
        #P=G(GtDG)^-1GtD -> (I-P)phi = phihat
        L = self.pvf_estimator.get_operator()
        n_states_actions = self.pvf_estimator.n_states_actions
        P = la.multi_dot([self.gradient_space.T, self.gradient_space, np.diag(self.weights)])
        #A norma 1????
        Lg = la.multi_dot([P, L, P.T])

        if np.allclose(self.L.T, self.L):
            eigval, eigvec = la.eigh(self.L)
        else:
            eigval, eigvec = la.eig(self.L)
            eigval, eigvec = abs(eigval), abs(eigvec)
            ind = eigval.argsort()[::-1]
            eigval, eigvec = eigval[ind], eigvec[ind]

        self.eigval = eigval
        self.eigvec = eigvec

    def _fit_remove_projections(self):
        eigval, eigvec = self.pvf_estimator.transform()
        eigvec_norm = la.multi_dot([eigvec.T, np.diag(self.weights), eigvec]).diagonal()
        coeff = la.multi_dot([eigvec.T, np.diag(self.weights), self.gradient_space]) / eigvec_norm
        projection = np.dot(coeff, eigval)

        gpvf = eigvec - projection
        rank = eigval / la.norm(gpvf, axis=0)
        ind = rank.argsort()
        if self.pvf_estimator.get_operator_type() == 'rand-walk':
            ind = ind[::-1]

        self.gpvf = gpvf[:, ind]
        self.rank = rank[ind]

    def _fit_remove_projections_and_orthogonalize(self):
        eigval, eigvec = self.pvf_estimator.transform()



