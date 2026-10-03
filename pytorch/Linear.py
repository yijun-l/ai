import numpy as np

from pytorch.Parameter import Parameter


class Linear:
    """Linear layer: y = x @ weight.T + bias

    weight: (out_features, in_features) = (V, D)
    bias:   (out_features,)             = (V, )

    In LM head,
    - weight = (V, D)
    - bias = (V,)
    - x = (B, T, D)
    - y = (B, T, V)
    """

    def __init__(self, in_features, out_features):
        rng = np.random.default_rng(seed=0)
        w_init = rng.standard_normal((out_features, in_features)).astype(np.float32)
        b_init = np.zeros(out_features, dtype=np.float32)

        self.weight = Parameter(w_init)
        self.bias = Parameter(b_init)
        self.input = None  # cached for backward

    def forward(self, input):
        """
        input: (..., in_features) -> (..., out_features)

        In LM head, (B, T, D) -> (B, T, V)
        """
        self.input = input
        return input @ self.weight.data.T + self.bias.data

    def backward(self, grad_output):
        """
        grad_output: (..., out_features); returns grad_input: (..., in_features)

        In LM head,
        - dLoss/dY = (B, T, V), out_features = V
        - dLoss/dX = (B, T, D), in_features = D
        """
        out_features, in_features = self.weight.shape

        x2 = self.input.reshape(-1, in_features)    # (N, D)
        g2 = grad_output.reshape(-1, out_features)  # (N, V)

        self.weight.grad += g2.T @ x2      # (out, in)
        self.bias.grad += g2.sum(axis=0)   # (out,)

        return (g2 @ self.weight.data).reshape(self.input.shape)

    def step(self, lr):
        self.weight.data -= lr * self.weight.grad
        self.bias.data -= lr * self.bias.grad
        self.weight.zero_grad()
        self.bias.zero_grad()


    def parameters(self):
        return [self.weight, self.bias]