import numpy as np
from numpy.random import uniform
from scipy.integrate import odeint

class Acrobot(object):
    """
    The Acrobot environment as presented in:
    "Tree-Based Batch Mode Reinforcement Learning, D. Ernst et. al."
    
    """
    def __init__(self):
        """
        Constructor.
        
        """
        # Properties
        self.state_dim = 4
        self.action_dim = 1        
        self.n_states = 0
        self.n_actions = 2        
        # State
        self.theta1 = uniform(-np.pi + 1, np.pi - 1)
        self.theta2 = self.dTheta1 = self.dTheta2 = 0.
        # End episode
        self.absorbing = False        
        # Constants
        self.g = 9.81
        self.M1 = self.M2 = 1
        self.L1 = self.L2 = 1
        self.mu1 = self.mu2 = .01
        # Time_step
        self.dt = .1        
        # Discount factor
        self.gamma = .95
    
    def step(self, u):
        """
        This function performs the step function of the environment.
        Args:
            u (int): the id of the action to be performed.            
        Returns:
            
            
        """
        stateAction = np.append([self.theta1,
                                 self.theta2,
                                 self.dTheta1,
                                 self.dTheta2], u)
        newState = odeint(self._dpds, stateAction, [0, self.dt])        
        
        newState = newState[-1]
        self.theta1 = newState[0]
        self.theta2 = newState[1]
        self.dTheta1 = newState[2]
        self.dTheta2 = newState[3]
        
        k = np.floor(self.theta1 / (2 * np.pi))
        x = np.array([self.theta1, self.theta2, self.dTheta1, self.dTheta2])
        value = self.theta1 - 2 * k * np.pi - np.pi
        o = np.array([value, 0., 0., 0.])
        d = np.linalg.norm(x - o)
        if(d < 1):
            self.absorbing = True
            return self.theta1, self.theta2, self.dTheta1, self.dTheta2, 1 - d
        else:
            return self.theta1, self.theta2, self.dTheta1, self.dTheta2, 0

    def reset(self, theta1, theta2, dTheta1, dTheta2):
        """
        This function set the position and velocity of the car to the
        starting conditions provided.
        Args:
            theta1 (float),
            theta2 (float),
            dTheta1 (float),
            dTheta2 (float)

        """
        self.absorbing = False
        self.theta1 = uniform(-np.pi + 1, np.pi - 1)
        self.theta2 = self.dTheta1 = self.dTheta2 = 0.

    def getState(self):
        """
        Returns:
            a tuple containing the state of the car represented by its position
            and velocity.
        
        """
        return [self.theta1, self.theta2, self.dTheta1, self.dTheta2]
        
    def isAbsorbing(self):
        """
        Returns:
            True if the state is absorbing, False otherwise.
        
        """
        return self.absorbing
        
    def _dpds(self, stateAction, t):
        theta1 = stateAction[0]
        theta2 = stateAction[1]
        dTheta1 = stateAction[2]
        dTheta2 = stateAction[3]
        action = stateAction[-1]

        d11 = self.M1 * self.L1 * self.L1 + self.M2 * (self.L1 * self.L1 +
            self.L2 * self.L2 + 2 * self.L1 * self.L2 * np.cos(theta2))
        d22 = self.M2 * self.L2 * self.L2
        d12 = self.M2 * (self.L2 * self.L2 + self.L1 * self.L2 *
            np.cos(theta2))
        c1 = -self.M2 * self.L1 * self.L2 * dTheta2 * (2 * dTheta1 + dTheta2 *
            np.sin(theta2))
        c2 = self.M2 * self.L1 * self.L2 * dTheta1 * dTheta1 * np.sin(theta2)
        phi1 = (self.M1 * self.L1 + self.M2 * self.L1) * self.g * \
            np.sin(theta1) + self.M2 * self.L2 * self.g * \
            np.sin(theta1 + theta2)
        phi2 = self.M2 * self.L2 * self.g * np.sin(theta1 + theta2)

        u = -5. if action == 0 else 5.
        
        diffTheta1 = dTheta1
        diffTheta2 = dTheta2
        d12d22 = d12 / d22
        diffDiffTheta1 = (-self.mu1 * dTheta1 - d12d22 * u + d12d22 *
                        self.mu2 * dTheta2 + d12d22 * c2 + d12d22 * phi2 -
                        c1 - phi1) / (d11 - (d12d22 * d12))
        diffDiffTheta2 = (u - self.mu2 * dTheta2 - d12 * diffDiffTheta1 -
                        c2 - phi2) / d22;
             
        return (diffTheta1, diffTheta2, diffDiffTheta1, diffDiffTheta2, 0.)
        
    def runEpisode(self, fqi):
        """
        This function runs an episode using the regressor in the provided
        object parameter.
        Params:
            fqi (object): an object containing the trained regressor
        Returns:
            - a tuple containing:
                - number of steps
                - J
                - a flag indicating if the goal state has been reached
            - sum of collected reward
        
        """
        J = 0
        t = 0
        test_succesful = 0
        rh = []
        while(t < 500 and not self.isAbsorbing()):
            state = self.getState()
            action, _ = fqi.predict(np.array(state))
            position, velocity, r = self.step(action)
            J += self.gamma ** t * r
            t += 1
            rh += [r]
            
            if r == 1:
                print('Goal reached')
                test_succesful = 1
    
        return (t, J, test_succesful), rh
        
    def evaluate(self, fqi):
        """
        This function evaluates the regressor in the provided object parameter.
        This way of evaluation is just one of many possible ones.
        Params:
            fqi (object): an object containing the trained regressor.
        Returns:
            a numpy array containing the average score obtained starting from
            289 different states
        
        """
        discRewards = np.zeros((289))
    
        counter = 0
        for i in range(-8, 9):
            for j in range(-8, 9):
                position = 0.125 * i
                velocity = 0.375 * j
                
                self.reset(position, velocity)
                tupla, rhistory = self.runEpisode(fqi)
            
                discRewards[counter] = tupla[1]
                counter += 1
                
        return np.mean(discRewards)