class SampleEstimator(object):
    '''
    Abstract class
    '''

    tol = 1e-24  # to avoid divisions by zero

    def get_P(self):
        return self.P

    def get_mu(self):
        return self.mu

    def get_d_s_mu(self):
        return self.d_s_mu

    def get_d_sa_mu(self):
        return self.d_sa_mu

    def get_d_ss(self):
        return self.d_ss

    def get_d_sasa(self):
        return self.d_sasa

    def get_d_ss_mu(self):
        return self.d_ss_mu

    def get_d_sasa_mu(self):
        return self.d_sasa_mu

    def get_count_sa(self):
        return self.count_sa

    def get_J(self):
        return self.J

    def compute_PVF_sa(self, k, operator='norm-laplacian', method='on-policy'):
        pass

    def compute_PVF_s(self, k, operator='norm-laplacian', method='on-policy'):
        pass