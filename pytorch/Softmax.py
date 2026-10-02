import numpy as np


class Softmax:
    """Inference-only: converts logits into probabilities.

    Shape convention (no flattening):
        (..., V) -> (..., V)   sums to 1 along the last axis
    where V = vocab_size.
    """

    @staticmethod
    def softmax(x):
        """Apply softmax along the last axis.

        Args:
            x: Input of shape (..., V).

        Returns:
            Probabilities of the same shape, summing to 1 along the last axis.
        """
        x = np.asarray(x, dtype=np.float32)

        # Subtract max for numerical stability.
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)