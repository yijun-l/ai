import numpy as np


def main():
    # Prepare the Dataset (y = 1 * X1 + 2 * X2 + 10)
    X = np.array([[1, 2],
                  [2, 5],
                  [3, 4],
                  [2, 5],
                  [2, 6]])

    W_real = np.array([[1, 2]])
    Y = X @ W_real.T + 10

    # Initialize Parameters
    np.random.seed(0)

    W = np.array([[np.random.random()],
                  [np.random.random()]])  # Shape: (2, 1)
    b = np.random.random()

    epochs = 3000
    lr = 0.02

    print(f'Initialization: w1 = {W[0,0]:.1f}, w2 = {W[1,0]:.1f}, b = {b:.1f}')

    # Full Training Loop
    for epoch in range(epochs):
        # Forward Propagation (Prediction)
        Y_hat = X @ W + b  # Shape: (5, 1)

        # Compute Residual Error & Loss (MSE)
        error = Y_hat - Y  # Shape: (5, 1)
        loss = np.mean(error ** 2)

        # Backpropagation (Gradient Computation via Chain Rule)
        # d(MSE)/dW = 2 * mean(error * X) -> Shape: (1, 2)
        # d(MSE)/db = 2 * mean(error)    -> Scalar
        dW = 2 * np.mean(error * X, axis=0, keepdims=True)
        db = 2 * np.mean(error)

        # Parameter Update
        W = W - dW.T * lr
        b = b - db * lr

        if epoch % 200 == 0:
            print(f'Epoch [{epoch:4d}/{epochs}] | w1: {W[0,0]:.1f} | w2: {W[1,0]:.1f} | b: {b:.1f} | loss: {loss:.4f}')

    print("-" * 65)
    print(f'Trained Model: Y = {W[0,0]:.1f} * X1 + {W[1,0]:.1f} * X2 + {b:.1f} | Final Loss: {loss:.6f}')


if __name__ == "__main__":
    main()