import numpy as np


class Perception(object):
    """perception classifier
    parameters
    eat:float; Learning rate (between 0.0 and 1.0)

    n_iter:int
    Passes over the training dataset

    Attributes

    w_: 1d-array
    Weights after fitting
    errors_: list
    Number of misclassifications in every epoch"""

    def __init__(self):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.
        X: {array-like}, shape=[n_samples, n_features]
        y: array-like, shape=[n_samples], target value
        Return: self: object"""

        self.w_ = np.zeros(1 + X.shape[1]) # add w_0
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
