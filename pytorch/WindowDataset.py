import numpy as np

class WindowDataset:
    """Build sliding-window samples from a long token sequence.

    For a long sequence `ids` and a window length `context_length`:
        X_ids[i] = ids[i : i + context_length]   # context (input)
        Y_ids[i] = ids[i + context_length]       # next token (label)
    """

    def __init__(self, context_length):
        """
        Args:
            context_length: Number of tokens in each input window.
        """
        self.context_length = context_length

    def build(self, ids):
        """Slice a long sequence into (input, label) pairs.

        Args:
            ids: 1-D integer array of token ids.

        Returns:
            X_ids: Array of shape (N, context_length), each row is a context window.
            Y_ids: Array of shape (N,), each element is the next token after the window.

        Raises:
            ValueError: If ids is not 1-D.
        """
        ids = np.asarray(ids)
        if ids.ndim != 1:
            raise ValueError(
                f"ids must be a 1-D vector, got shape={ids.shape}"
            )

        X_ids, Y_ids = [], []
        for i in range(len(ids) - self.context_length):
            X_ids.append(ids[i:i + self.context_length])
            Y_ids.append(ids[i + self.context_length])

        return np.array(X_ids), np.array(Y_ids)