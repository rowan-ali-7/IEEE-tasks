import numpy as np

class LogisticRegression:

    def __init__(self, iterations=1000, lr=0.01):
        #Initialize the Logistic Regression model
        #Parameters:
        #iterations : Number of training iterations
        #lr : Learning rate for gradient descent
        self.iterations = iterations
        self.lr = lr
        self.weights = None
        self.bias = 0
        
    def sigmoid(self, z):#Compute the Sigmoid
        return 1 / (1 + np.exp(-z))
  
    def _compute_cost(self, y, y_hat):#Compute the Binary Cross-Entropy (BCE) loss
        #Parameters:
        #y : True labels
        #y_hat : Predicted 
        m = len(y)
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        cost = -(1/m) * np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
        return cost
    
    def _compute_gradients(self, X, y, y_hat):
        m = len(y)
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)
        return dw, db
    """        
    Compute the gradients of the cost function with respect
    to the weights and bias.
    Parameters:
    X : Feature matrix.

        Returns:
            tuple: Gradient of weights (dw) and bias (db)."""
    def fit(self, X, y):#Train the Logistic Regression model using Gradient Descent
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.costs = []
        for _ in range(self.iterations):
            z = np.dot(X, self.weights) + self.bias
            y_hat = self.sigmoid(z)
            cost = self._compute_cost(y, y_hat)
            dw, db = self._compute_gradients(X, y, y_hat)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            self.costs.append(cost)
            
    def predict(self, X):# Predict binary class labels for input data
        z = np.dot(X, self.weights) + self.bias
        y_hat = self.sigmoid(z)
        return (y_hat >= 0.5).astype(int)
    
    def accuracy(self, y_true, y_pred):#Calculate the accuracy
        return np.mean(y_true == y_pred)