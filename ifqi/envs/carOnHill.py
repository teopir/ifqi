import numpy as np
from scipy.integrate import odeint

class CarOnHill(object):
    """
    The Car On Hill environment as presented in:
    "Tree-Based Batch Mode Reinforcement Learning, D. Ernst et. al."
    
    """
    def __init__(self):
        """
        Constructor.
        
        """
        # Properties
        self.state_dim = 2
        self.action_dim = 1        
        self.n_states = 0
        self.n_actions = 2        
        # State
        self.position = -0.5
        self.velocity = 0.        
        # End episode
        self.absorbing = False        
        # Constants
        self.g = 9.81
        self.m = 1        
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
            the new position and velocity of the car.
            
        """
        stateAction = np.append([self.position, self.velocity], u)
        newState = odeint(self._dpds, stateAction, [0, self.dt])        
        
        newState = newState[-1]
        self.position = newState[0]
        self.velocity = newState[1]
        
        if(self.position < -1 or np.abs(self.velocity) > 3):
            self.absorbing = True
            return self.position, self.velocity, -1
        elif(self.position > 1 and np.abs(self.velocity) <= 3):
            self.absorbing = True
            return self.position, self.velocity, 1
        else:
            return self.position, self.velocity, 0

    def reset(self, position=-0.5, velocity=0.):
        """
        This function set the position and velocity of the car to the
        starting conditions provided.
        Args:
            position (float): the initial position,
            velocity (float): the initial velocity.
            
        """
        self.absorbing=False
        self.position = position
        self.velocity = velocity

    def getState(self):
        """
        Returns:
            a tuple containing the state of the car represented by its position
            and velocity.
        
        """
        return [self.position, self.velocity]
        
    def isAbsorbing(self):
        """
        Returns:
            True if the state is absorbing, False otherwise.
        
        """
        return self.absorbing
        
    def _dpds(self, stateAction, t):
        position = stateAction[0]
        velocity = stateAction[1]
        action = stateAction[-1]

        if position < 0.:
            diffHill = 2 * position + 1
            diff2Hill = 2
        else:
            diffHill = 1 / ((1 + 5 * position ** 2) ** 1.5)
            diff2Hill = (-15 * position) / ((1 + 5 * position ** 2) ** 2.5)

        u = -4. if action == 0 else 4.
        
        dp = velocity
        ds = (u - self.g * self.m * diffHill - velocity ** 2 * self.m *
             diffHill * diff2Hill) / (self.m * (1 + diffHill ** 2))
             
        return (dp, ds, 0.)
        
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