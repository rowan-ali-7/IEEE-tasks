import numpy as np
import matplotlib.pyplot as plt
from classifi import *

X = np.array([
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    [1,0,1,0],
    [1,0,1,1],
    [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0],
    [1,1,1,1]
])

y = np.array([
    0,1,1,0,
    1,0,0,1,
    1,0,0,1,
    0,1,1,0
])
model = LogisticRegression(iterations=5000, lr=0.1)

model.fit(X, y)

predictions = model.predict(X)
print("Weights:", model.weights)
print("Bias:", model.bias)
print("True Labels :", y)
print("Predictions :", predictions)
print("Accuracy    :", model.accuracy(y, predictions))

"""
Creates the XOR dataset
Instantiates the LogisticRegression model
Trains the model
Predicts the class labels
Evaluates the model accuracy
"""