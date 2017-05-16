##############################################################
# Project 1. Build a ANN to Predict Daily Bike Rentals
# Zikri Bayraktar
# Last update: 02/19/2017
# Data is a time-series data for bike rentals over 2 years of
# time period. Last 21 days of the time period is used for
# testing, and 2 months before that used for verification.
##############################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data into dataframe via Pandas:
data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)

# plot some of the data:
rides[:24*10].plot(x='dteday', y='cnt')

# dummy variables:
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
  dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
  rides = pd.concat([rides, dummies], axis=1)

# remove some of the columns:
fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'weekday','atemp','mnth','workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()

# Scale variables to zero mean and stdev=1.
quant_features = ['casual', 'registered', 'cnt', 'temp','hum','windspeed']
# Store scalings in a dict to convert back later:
scaled_features={}
for each in quant_features:
  mean, std = data[each].mean(), data[each].std()
  scaled_features[each] = [mean,std]
  data.loc[:,each] = (data[each]-mean)/std

# Split data into training/testing/validaton sets:
# Save the last 21 days for testing:
test_data = data[-21*24:]
data = data[:-21*24] #remove the test_data from original set

# separate the data into features and targets:
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]  

# Split the validation data set (2 months data):
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]

# build the Neural Network:
# two layers: a hidden layer and an output layer.
# use sigmoid for hidden layer activation.
# use f(x)=x for output layer.
#---------------------------------------------------
class NeuralNetwork(object):
  def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
    # set number of nodes in input, hidden, output layers.
    self.input_nodes = input_nodes
    self.hidden_nodes = hidden_nodes
    self.output_nodes = output_nodes

    #Initialize weights:
    #it is important to set random weights small but not zero.
    self.w_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, (self.hidden_nodes, self.input_nodes))
    self.w_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, (self.output_nodes, self.hidden_nodes))
    self.lr = learning_rate
  
    # Sigmoid function:
    self.activation_function = lambda x: 1/(1+np.exp(-x))
  
    # Derivative of the sigmoid activation function:
    # f'(x) = sigmoid(x) *(1-sigmoid(x))
    self.activation_function_prime = lambda x: self.activation_function(x)*(1-self.activation_function(x))

  def train(self, inputs_list, targets_list):
    #convert inputs list to 2D array
    inputs = np.array(inputs_list, ndmin=2).T
    targets = np.array(targets_list, ndmin=2).T
	
	# NN Forward Pass:
	#Signals into the hidden layer:     (Win*x)
    hidden_inputs = np.dot(self.w_input_to_hidden, inputs)
	#Signals out of the hidden layer:    a = sigmoid(W*x)
    hidden_outputs = self.activation_function(hidden_inputs)
	
	#Signals into final output layer:    Wo*a
    final_inputs = np.dot(self.w_hidden_to_output, hidden_outputs)
    #signals out of the final layer:   f(Wo*a) = Wo*a 
    final_outputs = final_inputs
	
	# NN Backward Pass:
    output_errors = targets - final_outputs    # (y-y^)
    del_output_errors = output_errors * 1      # (y-y^)*f'(x) 

    #Backprop error:
    del_hidden_errors = np.dot(self.w_hidden_to_output.T, del_output_errors) * self.activation_function_prime(hidden_inputs)
    
	#update the weights:
    self.w_hidden_to_output += self.lr * del_output_errors*hidden_outputs.T
    self.w_input_to_hidden += self.lr * del_hidden_errors*inputs.T

  def run(self, inputs_list):
    #run Forward pass:
    inputs = np.array(inputs_list, ndmin=2).T
    hidden_inputs = np.dot(self.w_input_to_hidden, inputs)
    hidden_outputs = self.activation_function(hidden_inputs)

    final_inputs = np.dot(self.w_hidden_to_output, hidden_outputs)	
    final_outputs = final_inputs
    return final_outputs	
#---------------------------------------------------  

# Compute mean-squared_error:
def MSE(y,Y):
  return np.mean((y-Y)**2)

# Now set the hyperparameters and train the network:
import sys
# Hyperparameters:
epochs = 2000
learning_rate = 0.15
hidden_nodes = 24
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for e in range(epochs):
  #random batch of 128 records from training set:
  batch = np.random.choice(train_features.index, size=128)
  for record, target in zip(train_features.ix[batch].values, train_targets.ix[batch]['cnt']):
    network.train(record, target)
  
  #Print training progress:
  train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
  val_loss = MSE(network.run(val_features), val_targets['cnt'].values)  

  sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    
  losses['train'].append(train_loss)
  losses['validation'].append(val_loss)

  
# end-of-file