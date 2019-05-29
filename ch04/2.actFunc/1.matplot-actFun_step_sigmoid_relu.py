# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:04:33 2018

@author: shkim
"""
import numpy as np
import matplotlib.pyplot as plt

def step_func(x):
    return np.array(x>0, dtype=np.int)
	
def sigmoid_func(x):
    return 1/(1 + np.exp(-x))
	
def relu_func(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = step_func(x)
z = sigmoid_func(x)
plt.plot(x, y)
plt.plot(x, z)
plt.ylim(-0.1, 1.1)
plt.show()

x = np.arange(-6.0, 6.0, 0.1)
y = relu_func(x)
plt.plot(x, y)
plt.ylim(-1, 6.2)

plt.show()



