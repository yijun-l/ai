import numpy as np

from pytorch.Parameter import Parameter


class PositionEmbedding:
    """Learned absolute positional embedding, added to the input.

    Shapes:
        (..., T, D) -> (..., T, D)   with T <= context_length
    """

    def __init__(self, context_length, embed_dim, dtype=np.float32, seed=None):
        """
        Args:
            context_length: Max number of positions (C).
            embed_dim:      Dimension of each positional vector (D).
            dtype:          Data type of weight and gradients.
            seed:           Optional random seed.
        """
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.dtype = dtype

        rng = np.random.default_rng(seed)
        init = rng.standard_normal((context_length, embed_dim)).astype(dtype) * 0.02
        self.weight = Parameter(init)

        # Cache for backward.
        self._seq_len = None

    def forward(self, X):
        """Add the first T positional vectors to X.

        Args:
            X: Input of shape (..., T, D).

        Returns:
            Tensor of shape (..., T, D), same shape as X.
        """
        X = np.asarray(X, dtype=self.dtype)
        T = X.shape[-2]

        if T > self.context_length:
            raise ValueError(
                f"T={T} exceeds context_length={self.context_length}"
            )

        self._seq_len = T
        return X + self.weight.data[:T]

    def backward(self, dX):
        """Backward pass.

        dX passes through unchanged; dW[:T] = sum over leading dims.

        Args:
            dX: Upstream gradient of shape (..., T, D).

        Returns:
            Gradient w.r.t. input, shape (..., T, D).
        """
        if self._seq_len is None:
            raise RuntimeError("backward called before forward")

        dX = np.asarray(dX, dtype=self.dtype)
        T = self._seq_len

        if dX.ndim > 2:
            grad_T = dX.reshape(-1, T, self.embed_dim).sum(axis=0)
        else:
            grad_T = dX

        self.weight.grad[:T] += grad_T
        return dX

    def step(self, lr):
        """SGD update using the last computed gradient."""
        self.weight.data -= lr * self.weight.grad
        self.weight.zero_grad()

    def parameters(self):
        """Return the list of trainable parameters."""
        return [self.weight]