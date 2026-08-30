class SGD:

    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self):
        # Update model parameters using gradient descent
        self.model.w = self.model.w - self.lr * self.model.dw
        self.model.b = self.model.b - self.lr * self.model.db

    def zero_grad(self):
        # Reset gradients to zero
        self.model.dw = 0.0
        self.model.db = 0.0
