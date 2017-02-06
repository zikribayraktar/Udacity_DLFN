import numpy as np
from dataprep import features, targets, features_test, targets_test

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
a = targets.shape
c,d = features_test.shape
e= targets_test.shape
last_loss = None

print(n_records, n_features, a,c,d,e)
print(features.head(n=5))

# Initialize weights
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target
        
        h = np.dot(weights,x)
        # TODO: Calculate the output
        output = sigmoid(h)

        # TODO: Calculate the error
        error = y-output
        error_grad = error * sigmoid_prime(h)
        # TODO: Calculate change in weights
        del_w += error_grad * x

        # TODO: Update weights
    weights += learnrate *del_w/n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss


# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))