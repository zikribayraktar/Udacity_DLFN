import numpy as np
from dataprep import features, targets, features_test, targets_test

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Hyperparameters
n_hidden = 3  # number of hidden units
epochs = 500
learnrate = 0.1

n_records, n_features = features.shape
last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5, size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5, size=n_hidden)

print(weights_input_hidden.shape)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # TODO: Calculate the output
        h2in = np.dot(x, weights_input_hidden)
        a = sigmoid(h2in) 
        h2out = np.dot(a, weights_hidden_output)
        yh = sigmoid(h2out)

        ## Backward pass ##
        # TODO: Calculate the error
        error = y-yh

        # TODO: Calculate error gradient in output unit
        del_output_error = error * sigmoid_prime(h2out)
#        output_error = None

        # TODO: propagate errors to hidden layer
        del_hidden_error = np.dot(del_output_error, weights_hidden_output) * sigmoid_prime(h2in) 
#        hidden_error = None

        # TODO: Update the change in weights
        del_w_hidden_output += del_output_error * a
        del_w_input_hidden += del_hidden_error * x[:,None]

    # TODO: Update weights
    weights_input_hidden += learnrate * del_w_input_hidden/n_records
    weights_hidden_output += learnrate * del_w_hidden_output/n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_activations = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_activations,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))