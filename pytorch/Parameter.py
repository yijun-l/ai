import numpy as np


class Parameter:
    """Trainable parameter holding data and its gradient (like nn.Parameter)."""

    def __init__(self, data, dtype=np.float32):
        """
        Args:
            data:  Initial value, e.g. shape (V, D) or (C, D).
            dtype: Data type of data and grad.
        """
        data = np.asarray(data, dtype=dtype)
        self.data = data
        self.grad = np.zeros_like(data)

    def zero_grad(self):
        """Reset gradient to zeros."""
        self.grad[...] = 0.0

    @property
    def shape(self):
        """Shape of data and grad."""
        return self.data.shape