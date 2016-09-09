# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:25:48 2016

@author: samuele

Pendulum as described in Reinforcement Learning in Continuos Time and Space
"""

import numpy as np

class SwingPendulum:

    name = "swingUpPendulum"

    def __init__(self, **kwargs):
       self.m = 1.
       self.l = 1.
       self.g = 9.8
       self.mu = 0.01
       
       self.dt = 0.02
       self.state = [0.,0.]
       
       self.overTime = 0
       self.upTime = 0
       
       self.isGoal = False
       self.abs = False
       
       self.reset()



    def reset(self):
        theta = np.random.uniform(-np.pi, np.pi)
        self.state = [theta,0.]
        self.overTime = 0
        self.upTime = 0
       
        self.isGoal = False
        self.abs = False

    def step(self, u):
        theta, theta_dot = tuple(self.state)
       
        theta_ddot = (- self.mu * theta_dot + self.m * self.l * self.g * np.sin(theta_dot))/ (self.m * self.l)
       
        theta_dot += theta_ddot * self.dt
        theta += theta_dot * self.dt        
        
        self.state = [theta, theta_dot]
        
        if abs(theta) > 5*np.pi:
            self.overTime += 1
            if self.overTime * self.dt >= 1:
                self.abs = True
                return -1.
        else:
            self.overTime = 0
            if abs(theta) < np.pi/4.:
                self.upTime += 1
                if self.upTime * self.dt >= 10:
                    self.isGoal = True
                    return 1
            else:
                self.upTime = 0
                    
        return np.cos(theta)
        
       
    def isAbsorbing(self):
        return self.abs
        

    def isAtGoal(self):
        return self.isGoal

    def getState(self):
        return self.state
        

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
        horizon= 400
        while(t < horizon and not self.isAbsorbing()):
            state = self.getState()
            action, _ = fqi.predict(np.array(state))
            r = self.step(action)
            J += self.gamma ** t * r
            t += 1
            
            if r == 1:
                print('Goal reached')
                test_succesful = 1
    
        return t, J, test_succesful
        
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

                
        self.reset()
        J, step, goal = self.runEpisode(fqi)
               
        #(J, step, goal)
        return (J, step, goal)
        
