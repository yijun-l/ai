import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        # Initialize weight and bias with fixed random seed
        np.random.seed(0)

        self.W = np.random.random((in_features, out_features))
        self.b = np.random.random()

        # Initialize gradients for weight and bias

        self.dW = np.zeros((in_features, out_features))
        self.db = 0.0

        # Cache input tensor for backward pass
        self.X = None

    def forward(self, X):
        # Forward propagation: compute predictions (Y_hat = W * X + B)
        self.X = X
        Y_pred = X @ self.W + self.b

        return Y_pred

    def backward(self, dY_pred):
        """
        Backpropagation: compute gradients with respect to w, b, and x
        Args:
            dY_pred: Upstream gradient (dLoss / dY_pred)
        """
        # Compute gradients for trainable parameters (dw, db)
        self.dW = np.mean(dY_pred * self.X, axis=0, keepdims=True).T
        self.db = np.mean(dY_pred)

        # Compute gradient with respect to input (dx) to pass upstream
        dX = np.mean(dY_pred) * self.W
        return dX
