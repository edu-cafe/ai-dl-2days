# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 23:44:23 2019

@author: shkim
"""

#%%
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *

#%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#%load_ext autoreload
#%autoreload 2

np.random.seed(1)

#%%
#from dnn_app_utils_v3 import *

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

#%%

# Example of a picture
index = 10
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

#%%

# Explore your dataset 
print('train_x_org_shape : ' + str(train_x_orig.shape))   # (209, 64, 64, 3)
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))  # 209
print ("Number of testing examples: " + str(m_test))  # 50
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)") #(64, 64, 3)
print ("train_x_orig shape: " + str(train_x_orig.shape))  # (209, 64, 64, 3)
print ("train_y shape: " + str(train_y.shape))  # (1, 209)
print ("test_x_orig shape: " + str(test_x_orig.shape))  # (50, 64, 64, 3)
print ("test_y shape: " + str(test_y.shape))  # (1, 50)

#%%

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))  # (12288, 209)
print ("test_x's shape: " + str(test_x.shape))  # (12288, 50)

#%%
### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model

#%%

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

#%%
# DNN 모델로 학습시키고 cost 출력
...........

#%%
# training-set accuracy 출력
..............

#%%
# test-set accuracy 출력
..............

#%%
# 잘못 분류된 이미지 정보 출력
.......................

#%%
  