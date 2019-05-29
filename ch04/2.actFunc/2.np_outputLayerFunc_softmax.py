# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:56:58 2018

@author: shkim
"""

import numpy as np

a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a/sum_exp_a
print(y)


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y

a = np.array([0.3, 2.9, 4.0])
print(softmax(a))    # [0.01821127 0.24519181 0.73659691]

#print(sum(softmax(a)))

# -------------------------------
# nan example
a = np.array([1010, 1000, 990])
print(np.exp(a)/np.sum(np.exp(a)))
c = np.max(a)
print(a -c)
print(np.exp(a-c)/np.sum(np.exp(a-c)))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)    # [0.01821127 0.24519181 0.73659691]
print(np.sum(y))

a = np.array([1010, 1000, 990])
y = softmax(a)
print(y)    # [9.99954600e-01 4.53978686e-05 2.06106005e-09]
print(np.sum(y))






