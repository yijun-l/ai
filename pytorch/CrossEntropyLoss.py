import numpy as np

class CrossEntropyLoss:
    """Training-only: takes logits, performs log_softmax + NLL internally."""

    def __init__(self):
        self.targets = None
        self.log_probs = None

    def forward(self, logits, targets):
        """Compute the mean cross-entropy loss from raw logits.

        Args:
            logits:  Raw scores of shape (N, V).
            targets: Ground-truth class indices of shape (N,).

        Returns:
            Scalar loss.
        """
        self.targets = targets
        N, V = logits.shape

        # Subtract max for numerical stability.
        x = logits - logits.max(axis=-1, keepdims=True)
        logsumexp = np.log(np.exp(x).sum(axis=-1, keepdims=True))
        self.log_probs = x - logsumexp                      # log_softmax

        loss = -self.log_probs[np.arange(N), targets].mean()
        return loss

    def backward(self):
        """Backward pass.

        Returns:
            Gradient w.r.t. logits, shape (N, V).
        """
        if self.log_probs is None:
            raise RuntimeError("backward called before forward")

        N, V = self.log_probs.shape
        grad = np.exp(self.log_probs)          # equivalent to probs
        grad[np.arange(N), self.targets] -= 1  # subtract one-hot
        grad /= N
        return grad                            # gradient w.r.t. logits