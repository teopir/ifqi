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
       theta = np.random.uniform(-np.pi, np.pi)
       self.state = [theta,0.]
       
       self.overTime = 0
       self.upTime = 0
       
       self.isGoal = False
       self.abs = False
       
       self.state_dim = 2
       self.action_dim = 1
       self.n_actions = 11
       self.gamma = 0.9
       self.previusTheta = 0
       
       self.overRotated = False
       self.cumulatedRotation = 0 
       self.overRotatedTime = 0



    def reset(self):
        theta = np.random.uniform(-np.pi, np.pi)
        self.state = [theta,0.]
        self.overTime = 0
        self.upTime = 0
       
        self.isGoal = False
        self.abs = False
        self.previusTheta = 0
        self.overRotated = False
        self.overRotatedTime =0
        self.cumulatedRotation = 0
        
    def step(self, action):
        
        u = action[0]  / 11. * 10. - 5.
        
        theta, theta_dot = tuple(self.state)
       
        #theta_ddot = (- self.mu * theta_dot + self.m * self.l * self.g * np.sin(theta_dot) + u)/ (self.m * self.l * self.l)
        theta_ddot = (- self.dt * theta_dot + self.m * self.l * self.g * np.sin(theta_dot) + u)#/ (self.m * self.l * self.l)

        #bund theta_dot
        theta_dot_temp = theta_dot + theta_ddot        
        if theta_dot_temp > np.pi/self.dt:
            theta_dot_temp = np.pi/self.dt
        if theta_dot_temp < -np.pi/self.dt:
            theta_dot_temp = -np.pi/self.dt
        
        theta_dot = theta_dot_temp#theta_ddot #* self.dt
        theta += theta_dot * self.dt        
        
        #Adjust Theta        
        if theta > np.pi:
            theta -= 2*np.pi
        if theta < -np.pi:
            theta += 2*np.pi
            
        self.state = [theta, theta_dot]
        
        """signAngleDifference = np.arctan2(np.sin(theta - self.previusTheta), np.sin(theta - self.previusTheta))
        self.cumulatedRotation += signAngleDifference
        
        if (not self.overRotated and abs(self.cumulatedRotation) > 5.0 * np.pi):
            self.overRotated = True
        if (self.overRotated):
            self.overRotatedTime += 1
        
        self.abs = self.overRotated and (self.overRotatedTime > 1.0 / self.dt)"""
    
    
        """if not self.overRotated:
            return np.cos(theta)
        else:
            return -1"""
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
        J = []
        t = 0
        test_succesful = 0
        horizon= 400
        while(t < horizon and not self.isAbsorbing()):
            state = self.getState()
            action, _ = fqi.predict(np.array(state))
            r = self.step(action)
            #J += self.gamma ** t * r
            J.append(r)
            t += 1
            
            if r == 1:
                print('Goal reached')
                test_succesful = 1
        return t, np.mean(J), test_succesful
        
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

        step, J, goal = self.runEpisode(fqi)

        print "step", step
        print "J", J
        #(J, step, goal)
        return (J, step, goal)
        
