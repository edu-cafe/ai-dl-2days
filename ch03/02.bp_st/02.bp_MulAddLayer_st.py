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
class AddLayer:
    def __init__(self):
        pass
        
    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1       
        return dx, dy

#%%  
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layers
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = .............
price = mul_tax_layer.forward(all_price, tax)
print('price: ', price)    # price:  715.0000000000001

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = ............
print('dapple:', dapple, ', dapple_num:', dapple_num)  # dapple: 2.2 , dapple_num: 110.00000000000001
print('dorange:', dorange, ', dorange_num:', dorange_num)  # dorange: 3.3000000000000003 , dorange_num: 165.0
print('dall_price:', dall_price, ', dtax:', dtax)  # dall_price: 1.1 , dtax: 650






