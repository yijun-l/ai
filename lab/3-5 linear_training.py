# test_linear.py
import numpy as np

from pytorch.Linear import Linear


# Ground truth
# y0 = 1*x0 + 2*x1 + 0.5
# y1 = 3*x0 + 4*x1 - 1.0
# y2 = 5*x0 + 6*x1 + 2.0
W_TRUE = np.array([[1., 2.],
                   [3., 4.],
                   [5., 6.]], dtype=np.float32)   # (V, D) = (3, 2)
B_TRUE = np.array([0.5, -1.0, 2.0], dtype=np.float32)  # (V,) = (3,)


def make_data(n, rng):
    """Generate y = x @ W_TRUE.T + B_TRUE."""
    x = rng.standard_normal((n, 2)).astype(np.float32)   # (n, 2)
    y = x @ W_TRUE.T + B_TRUE                            # (n, 3)
    return x, y


def mse_loss(y_pred, y_true):
    """MSE loss and its gradient w.r.t. y_pred."""
    diff = y_pred - y_true
    loss = np.mean(diff ** 2)
    grad = 2.0 * diff / y_true.size    # dL/dy_pred
    return loss, grad


def train():
    rng = np.random.default_rng(seed=0)

    D, V = 2, 3
    N = 200
    lr = 0.1
    epochs = 500

    x, y = make_data(N, rng)
    layer = Linear(D, V)

    for epoch in range(epochs):
        y_pred = layer.forward(x)
        loss, grad = mse_loss(y_pred, y)
        layer.backward(grad)
        layer.step(lr)

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  loss {loss:.3f}")

    # Compare learned parameters against ground truth
    print("\n--- learned vs true ---")
    print("W_true:\n", W_TRUE)
    print("W_learned:\n", layer.weight.data)
    print("\nb_true:   ", B_TRUE)
    print("b_learned:", layer.bias.data)

    # Sanity check: with x = [1, 0], output should be W_TRUE[:, 0] + B_TRUE
    x0 = np.array([[1.0, 0.0]], dtype=np.float32)
    y0_true = x0 @ W_TRUE.T + B_TRUE
    y0_hat = layer.forward(x0)
    print("\n--- sanity check (x = [1, 0]) ---")
    print("true:", y0_true)   # expected [1.5, 2.0, 7.0]
    print("pred:", y0_hat)

    # Evaluate on unseen data
    x_test, y_test = make_data(50, rng)
    y_hat = layer.forward(x_test)
    test_mse = np.mean((y_hat - y_test) ** 2)
    print("\ntest MSE:", test_mse)


if __name__ == "__main__":
    train()