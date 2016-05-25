import interface as bbox

# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:47:34 2016

@author: samuele
"""
import numpy as np

class BlackBox(object):
    
    def prepare_bbox(self):
        if bbox.is_level_loaded():
            bbox.reset_level()
        else:
            bbox.load_level("dataset/blackbox_data/levels/train_level.data", verbose=1)
        
    def __init__(self):
        self.hasNext=True
        self.prepare_bbox()
        self.score = bbox.get_score()
    
    def step(self, u):
        self.hasNext = bbox.do_action(u)
        ret = bbox.get_score() - self.score 
        self.score = bbox.get_score() 
        return ret
        

    def reset(self):
        self.prepare_bbox()
        
    def getState(self):
        return bbox.get_state()
        
    def isAbsorbing(self):
        return not self.hasNext