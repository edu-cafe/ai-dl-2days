# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:58:00 2018

@author: shkim
"""

import numpy as np

#%%
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x        
        return dx, dy

#%%  
apple = 100
apple_num = 2
tax = 1.1

# layers
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = .........
price = ........
print('price: ', price)    # price:  220.00000000000003

# backward
dprice = 1
dapple_price, dtax = ..........
dapple, dapple_num = ..........
print('dapple:', dapple, ', dapple_num:', dapple_num, ', dtax:', dtax)
# dapple: 2.2 , dapple_num: 110.00000000000001 , dtax: 200





