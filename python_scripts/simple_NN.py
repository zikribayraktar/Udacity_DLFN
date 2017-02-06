# Simple NN script
import numpy as np

#-------------------------------------------------
# Defining the sigmoid activation function
# f(x) = 1/(1+exp(x))
def sigmoid(x):
    return 1/(1+np.exp(-x))
	
#-------------------------------------------------
# Derivative of the sigmoid function
# df(x)/dx = f(x) * (1-f(x))  where f(x) is sigmoid function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
	
#-------------------------------------------------

x = np.array([0.1, 0.3])    #inputs
y = 0.2						#output
w = np.array([-0.8, 0.5])	#weights

# The learning rate, eta in the weight step equation
learnrate = 0.5

# The neural network unit input h:
h = np.dot(x,w)
# The neural network output:
nn_output = sigmoid(h)

# output error
error = y - nn_output

# dE/dw = (y-h) * f'(h) * x
# but 
# error gradient = (y-h) * f'(h)
error_grad = error * sigmoid_prime(h)

# Gradient descent step
# del_w = eta * error_grad
del_w = learnrate * error_grad * x