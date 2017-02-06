# Predicting Boston Housing Prices
# Python 3.5 on Windows 10
# Using Sklearn Linear Regression models
# Zikri Bayraktar
# On Command Line:  >> python housing.py
#--------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

# Load the Boston Housing data set:
BostonData = load_boston()
# Print the Shape of the Sklearn Dataset
print(BostonData.data.shape)
x = BostonData['data']
y = BostonData['target']

# Make and fit linear regression model with multiple independent variables
model = LinearRegression()
model.fit(x,y)


# Make a prediction using the model
sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
# TODO: Predict housing price for the sample_house
prediction = model.predict(sample_house)
print(prediction)
