import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.array([0.5, 0.1, -0.2])
y = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
h2in = np.dot(x, weights_input_hidden)
a = sigmoid(h2in)

h2out = np.dot(a, weights_hidden_output)
yh = sigmoid(h2out)

## Backwards pass
## TODO: Calculate error
error = y - yh

# TODO: Calculate error gradient for output layer
del_err_output = error * sigmoid_prime(h2out)

# TODO: Calculate error gradient for hidden layer
del_err_hidden = np.dot(del_err_output, weights_hidden_output) * sigmoid_prime(h2in) 

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * del_err_output * a

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_o = learnrate * del_err_hidden * x[:, None]

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_o)
