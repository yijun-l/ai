import numpy as np


class Linear:
    def __init__(self):
        # Initialize weight and bias with fixed random seed
        np.random.seed(0)

        self.w = np.random.random()
        self.b = np.random.random()

        # Initialize gradients for weight and bias

        self.dw = 0.0
        self.db = 0.0

        # Cache input tensor for backward pass
        self.x = None

    def forward(self, x):
        # Forward propagation: compute predictions (Y_hat = W * X + B)
        self.x = x
        y_pred = x * self.w + self.b

        return y_pred

    def backward(self, dy_pred):
        """
        Backpropagation: compute gradients with respect to w, b, and x
        Args:
            dy_pred: Upstream gradient (dLoss / dy_pred)
        """
        # Compute gradients for trainable parameters (dw, db)
        self.dw = np.mean(dy_pred * self.x)
        self.db = np.mean(dy_pred)

        # Compute gradient with respect to input (dx) to pass upstream
        dx = dy_pred * self.w
        return dx
