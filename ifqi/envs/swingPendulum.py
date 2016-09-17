# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:25:48 2016

@author: samuele

Pendulum as described in Reinforcement Learning in Continuos Time and Space
"""

import numpy as np

from environment import Environment

class SwingPendulum(Environment):

    name = "swingUpPendulum"

    def __init__(self, **kwargs):
       self.state_dim = 2
       self.action_dim = 1
       self.n_actions = 11
       self.gamma = 0.9        
        
       self._m = 1.
       self._l = 1.
       self._g = 9.8
       self._mu = 0.01
       
       self._dt = 0.02
       theta = np.random.uniform(-np.pi, np.pi)
       self._state = [theta,0.]
       
       self._overTime = 0
       self._upTime = 0
       
       self._atGoal = False
       self._abs = False
       
       self._previousTheta = 0
       
       self._overRotated = False
       self._cumulatedRotation = 0 
       self._overRotatedTime = 0

    def _reset(self, state=None):
        theta = np.random.uniform(-np.pi, np.pi)
        self._state = [theta,0.]
        self._overTime = 0
        self._upTime = 0
       
        self._atGoal = False
        self._abs = False
        self._previousTheta = 0
        self._overRotated = False
        self._overRotatedTime =0
        self._cumulatedRotation = 0
        
    def _step(self, action, render=False):
        
        u = action[0]  / 11. * 10. - 5.
        
        theta, theta_dot = tuple(self._state)
       
        #theta_ddot = (- self.mu * theta_dot + self.m * self.l * self.g * np.sin(theta_dot) + u)/ (self.m * self.l * self.l)
        theta_ddot = (- self._dt * theta_dot + self._m * self._l * self._g * np.sin(theta_dot) + u)#/ (self.m * self.l * self.l)

        #bund theta_dot
        theta_dot_temp = theta_dot + theta_ddot        
        if theta_dot_temp > np.pi/self._dt:
            theta_dot_temp = np.pi/self._dt
        if theta_dot_temp < -np.pi/self._dt:
            theta_dot_temp = -np.pi/self._dt
        
        theta_dot = theta_dot_temp#theta_ddot #* self.dt
        theta += theta_dot * self._dt        
        
        #Adjust Theta        
        if theta > np.pi:
            theta -= 2*np.pi
        if theta < -np.pi:
            theta += 2*np.pi
            
        self._state = [theta, theta_dot]
        
        """signAngleDifference = np.arctan2(np.sin(theta - self.previousTheta), np.sin(theta - self.previousTheta))
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

    def _getState(self):
        return self.state
        
    def evaluate(self, fqi, expReplay=False, render=False):
        """
        This function evaluates the regressor in the provided object parameter.
        This way of evaluation is just one of many possible ones.
        Params:
            fqi (object): an object containing the trained regressor.
        Returns:
            a numpy array containing the average score obtained starting from
            289 different states
        
        """
        self._reset()

        step, J, goal = self._runEpisode(fqi, expReplay, render)

        print "step", step
        print "J", J
        #(J, step, goal)
        return (J, step, goal)
        
