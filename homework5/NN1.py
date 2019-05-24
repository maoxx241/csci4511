# 1 layer feed forward network
# from https://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np
import matplotlib.pyplot as plt
# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset. Each row is a sample, columns correspond to input nodes.
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation deterministic (a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0. Array is 3x1 since input
# data are vectors of 3 numbers. 
syn0 = 2*np.random.random((3,1)) - 1

for iter in range(1000):
    
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1
    
    # print('l1 error ',l1)

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print("Output After Training:")
print(l1)
print("final weights")
print(syn0)

#l1= nonlin(np.dot(l0,syn0))
k0=nonlin(np.array([1,2,1]).dot(syn0))
k1=nonlin(np.array([0,4,1]).dot(syn0))
k2=nonlin(np.array([5,2,1]).dot(syn0))
k3=nonlin(np.array([-2,4,1]).dot(syn0))
l1_error = y - k0
print("Output For 3.3:")
print(k0)
print(k1)
print(k2)
print(k3)

