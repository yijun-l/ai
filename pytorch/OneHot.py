import numpy as np


class OneHot:
    """Encode token ids into one-hot vectors.

    Shape convention (consistent, no flattening):
        (T,)    -> (T, V)
        (B, T)  -> (B, T, V)
    where V = vocab_size.
    """

    def __init__(self, vocab_size, dtype=np.float32):
        """
        Args:
            vocab_size: Number of distinct tokens (size of the vocabulary).
            dtype: Data type of the output one-hot array. Defaults to float32.
        """
        self.vocab_size = vocab_size
        self.dtype = dtype

    # ---------- Forward pass ----------
    def forward(self, ids):
        """Convert token ids into one-hot vectors.

        Args:
            ids: 1-D or 2-D integer array of token ids.
                 Values must be in [0, vocab_size).

        Returns:
            One-hot array of shape (T, V) if ids is 1-D,
            or (B, T, V) if ids is 2-D.

        Raises:
            ValueError: If ids is not 1-D or 2-D.
        """
        ids = np.asarray(ids)
        if ids.ndim not in (1, 2):
            raise ValueError(
                f"ids must be a 1-D or 2-D array, got shape={ids.shape}"
            )

        eye = np.eye(self.vocab_size, dtype=self.dtype)
        return eye[ids]          # Fancy indexing: replace each id with its one-hot row

    # ---------- Backward pass ----------
    def backward(self, dX):
        """Backward pass of the one-hot layer.

        This layer has no trainable parameters, so the gradient is simply
        passed through to the upstream layer unchanged.

        Args:
            dX: Upstream gradient, same shape as the forward output.

        Returns:
            dX unchanged.
        """
        return dX