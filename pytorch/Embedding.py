import numpy as np


class Embedding:
    """Embedding layer whose forward input is a one-hot tensor.

    Shape convention (consistent with OneHot, no flattening):
        (..., V)  ->  (..., D)
    i.e.
        (T, V)     -> (T, D)
        (B, T, V)  -> (B, T, D)
    where V = vocab_size, D = embed_dim.

    Forward:
        y = x_one_hot @ weight

    This is equivalent to gathering rows by token id, but here the
    selection is done by a matmul with the one-hot matrix.
    """

    def __init__(self, vocab_size, embed_dim, dtype=np.float32, seed=None):
        """
        Args:
            vocab_size: Size of the vocabulary (V), must match OneHot's vocab_size.
            embed_dim:  Dimension of each embedding vector (D).
            dtype:      Data type of weight and gradients. Defaults to float32.
            seed:       Optional random seed for reproducible initialization.
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dtype = dtype

        rng = np.random.default_rng(seed)
        self.weight = rng.standard_normal((vocab_size, embed_dim)).astype(dtype) * 0.02

        # Reusable gradient buffer.
        self.dE = np.zeros_like(self.weight)

        # Cache for backward.
        self._x_one_hot = None

    # ---------- Forward pass ----------
    def forward(self, x_one_hot):
        """Project one-hot vectors into embedding space.

        Args:
            x_one_hot: One-hot tensor of shape (..., V).
                       Typically (B, T, V) or (T, V).

        Returns:
            Embedding tensor of shape (..., D), e.g. (B, T, D) or (T, D).

        Raises:
            ValueError: If the last dimension does not equal vocab_size.
        """
        x_one_hot = np.asarray(x_one_hot, dtype=self.dtype)

        if x_one_hot.shape[-1] != self.vocab_size:
            raise ValueError(
                f"last dim of x_one_hot must be vocab_size={self.vocab_size}, "
                f"got shape={x_one_hot.shape}"
            )

        self._x_one_hot = x_one_hot
        return x_one_hot @ self.weight      # (..., V) @ (V, D) = (..., D)

    # ---------- Backward pass ----------
    def backward(self, dX):
        """Backward pass of the embedding layer.

        Forward:  Y     = X @ W          # X: one-hot, shape (..., V)
        Backward: dW    = X.T @ dY       # reduce over all leading dims

        With X being one-hot, X.T @ dY is exactly a scatter-add of dY
        into the rows selected by the one-hot positions.

        Args:
            dX: Upstream gradient, shape (..., D), same as forward output.

        Returns:
            dweight: Gradient w.r.t. self.weight, shape (V, D).

        Raises:
            RuntimeError: If forward was not called before backward.
        """
        if self._x_one_hot is None:
            raise RuntimeError("backward called before forward")

        dX = np.asarray(dX, dtype=self.dtype)

        # Flatten leading dims: (..., V) -> (N, V), (..., D) -> (N, D)
        X2 = self._x_one_hot.reshape(-1, self.vocab_size)
        dY2 = dX.reshape(-1, self.embed_dim)

        # dW = X.T @ dY   ->  (V, N) @ (N, D) = (V, D)
        self.dE = X2.T @ dY2

        return self.dE

    # ---------- Convenience: parameter update ----------
    def step(self, lr):
        """Simple SGD update using the last computed gradient."""
        self.weight -= lr * self.dE