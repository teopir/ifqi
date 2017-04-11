class ProjectionIRL(object):

    def __init__(self):
        pass

    def fit(self, theta0):
        #Compute expert feature expectation
        mu_expert =

        theta = theta0
        self.policy.set_parameter(theta)

        #Estimate feature expectation
        mu =

        #Compute mu bar
        mu_bar =

        #Compute weights
        w = mu_expert - mu_bar

        #Compute margin
        t = la.norm(w)

        if t < eps:
            break

        best = np.dot(reward_features, w)
        theta_best = #optimize(best)


