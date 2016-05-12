# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:47:34 2016

@author: samuele
"""
import numpy as np

class MountainCar(object):
    #state
    position = -0.5
    velocity = 0.
    
    #end episode
    absorbing = False
    
    #constants
    g = 9.81
    m = 1
    
    #time_step
    dt = .1
    
    #discount factor
    gamma = .95
    
    def step(self, u):
        
        if self.position < 0.:
            diffHill = 2 * self.position + 1
            diff2Hill = 2
        else:
            diffHill = 1 / ((1 + 5 * self.position ** 2) ** 1.5)
            diff2Hill = (-15 * self.position) / ((1 + 5 * self.position ** 2) ** 2.5)
        act = -4. if u == 0 else 4.
        acc = (act - self.g * self.m * diffHill - self.velocity ** 2 * self.m *
                diffHill * diff2Hill) / (self.m * (1 + diffHill ** 2))
    
        self.position = self.position + self.dt * self.velocity + 0.5 * acc * self.dt ** 2
        self.velocity = self.velocity + self.dt * acc
        
        if(self.position < -1 or np.abs(self.velocity) > 3):
            self.absorbing = True
            return self.position, self.velocity, -1
        elif(self.position > 1 and np.abs(self.velocity) <= 3):
            self.absorbing = True
            return self.position, self.velocity, 1
        else:
            return self.position, self.velocity, 0

    def reset(self, position=-0.5, velocity=0.):
        
        self.absorbing=False
        self.position = position
        self.velocity = velocity
        
    def getState(self):
        return [self.position, self.velocity]
        
    def isAbsorbing(self):
        return self.absorbing