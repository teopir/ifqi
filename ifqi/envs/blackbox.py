import interface as bbox

# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:47:34 2016

@author: samuele
"""
import numpy as np

class BlackBox(object):
    
    def prepare_bbox(self):
        self.next=True
        if bbox.is_level_loaded():
            bbox.reset_level()
        else:
            bbox.load_level("dataset/blackbox_data/levels/" + self.data_file + "_level.data", verbose=1)
        
    def __init__(self, data_file="train"):
        self.next = True
        self.data_file = data_file
        self.prepare_bbox()
        self.score = bbox.get_score()
    
    def step(self, u):
        self.next = bbox.do_action(u)
        ret = bbox.get_score() - self.score 
        self.score = bbox.get_score() 
        #if bbox.get_time() % 10000 == 0:
        #    print ("time = %d, score = %f" % (bbox.get_time(), bbox.get_score()))
        return ret
    
    def end(self):
        bbox.finish(verbose=1)
        
    def get_score(self):
        return bbox.get_score()
        
    def get_time(self):
        return bbox.get_time()


    def reset(self):
        self.prepare_bbox()
        
    def getState(self):
        return bbox.get_state()
        
    def hasNext(self):
        return self.next