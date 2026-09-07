class SGD:

    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self):
        # Update model parameters using gradient descent
        self.model.W = self.model.W - self.lr * self.model.dW
        self.model.b = self.model.b - self.lr * self.model.db

    def zero_grad(self):
        # Reset gradients to zero
        self.model.dW.fill(0.0)
        self.model.db = 0.0
