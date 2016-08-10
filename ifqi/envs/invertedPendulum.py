import numpy as np

class InvPendulum(object):
    """
    The Inverted Pendulum environment.
    
    """
    def __init__(self):
        """
        Constructor.
        
        """
        # State
        self.theta = 0.
        self.theta_dot = 0.
        # End episode
        self.absorbing=False
        # Constants
        self.g = 9.8
        self.m = 2.
        self.M = 8.
        self.l = .5
        self.alpha = 1. / (self.m + self.M)
        self.noise = 10.
        self.angleMax = np.pi / 2.
        # Time_step
        self.dt = 0.1
        # Discount factor
        self.gamma=.95
    
    def step(self, u):
        """
        This function performs the step function of the environment.
        Args:
            u (int): the id of the action to be performed.            
        Returns:
            the new position and velocity of the car.
            
        """
        act = u * 50. - 50.
        n_u = act  +  2 * self.noise * np.random.rand() - self.noise
        
        a = self.g * np.sin(self.theta) - self.alpha * self.m * self.l * self.theta_dot**2 * np.sin(2 * self.theta) / 2. - self.alpha * np.cos(self.theta) * n_u
        b = 4. *self.l / 3. - self.alpha * self.m * self.l * (np.cos(self.theta)) ** 2
        
        theta_ddot = a/b
    
        self.theta_dot = self.theta_dot + self.dt * theta_ddot
        self.theta = self.theta + self.dt * self.theta_dot
        
        if(np.abs(self.theta) > self.angleMax):
            self.absorbing = True
            return -1
        else:
            return 0

    def reset(self):
        """
        This function set the angle and angular velocity of the pendulum to the
        starting conditions.
            
        """
        self.absorbing=False
        self.theta = 0
        self.theta_dot = 0
        
    def getState(self):
        """
        Returns:
            a tuple containing the state of the pendulum represented by its angle
            and angular velocity
        
        """
        return [self.theta, self.theta_dot]
        
    def isAbsorbing(self):
        """
        Returns:
            True if the state is absorbing, False otherwise.
        
        """
        return self.absorbing
        
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
        while not self.isAbsorbing():
            state = self.getState()
            action, _ = fqi.predict(np.array(state))
            r = self.step(action)
            J += self.gamma**t * r
            if t > 3000:
                print('COOL! You done it!')
                test_succesful = 1
                break
            t += 1
            rh += [r]
        return (t, J, test_succesful), rh
        
    def evaluate(self, fqi):
        """
        This function evaluates the regressor in the provided object parameter.
        This way of evaluation is just one of many possible ones.
        Params:
            fqi (object): an object containing the trained regressor.
        Returns:
            ...
        
        """
        pass