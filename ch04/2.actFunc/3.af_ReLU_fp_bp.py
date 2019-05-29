# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:26:31 2018

@author: shkim
"""
import numpy as np

class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
        


