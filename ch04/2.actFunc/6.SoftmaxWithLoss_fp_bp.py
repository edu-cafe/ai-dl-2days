# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:26:31 2018

@author: shkim
"""
import numpy as np

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y

# CEE, Cross Entropy Error
def cee(p, t):	
    delta = 1e-7
    return -np.sum(t * np.log(p + delta) )  # delta : -log0이 -inf가 되는 것을 방지
	
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cee(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
      
#%%
