# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:47:34 2016

@author: samuele
"""
import numpy as np

class InvPendulum(object):
    #state
    theta = 0.
    theta_dot = 0.
    
    #end episode
    absorbing=False
    
    #constants
    g = 9.8
    m = 2.
    M = 8.
    l = .5
    alpha = 1. / (m + M)
    noise = 10.
    angleMax = np.pi / 2.
    
    #time_step
    dt = 0.1
    
    #discount factor
    gamma=.95
    
    def step(self, u):
        
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
        
        self.absorbing=False
        self.theta = 0
        self.theta_dot = 0
        
    def getState(self):
        return [self.theta, self.theta_dot]
        
    def isAbsorbing(self):
        return self.absorbing