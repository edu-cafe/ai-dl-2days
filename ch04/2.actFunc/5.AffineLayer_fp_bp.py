# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:26:31 2018

@author: shkim
"""
import numpy as np

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        print('fp_x shape: ', x.shape)
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        print('fp_x shape: ', x.shape)
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    
    def backward(self, dout):        
        dx = np.dot(dout, self.W.T)
        print('bp_npdot_dx:', dx.shape)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        print('self.original_x_shape:', self.original_x_shape)
        print('*self.original_x_shape:', *self.original_x_shape)
        dx = dx.reshape(*self.original_x_shape)
        print('bp_reshape_dx:', dx.shape)
        return dx

