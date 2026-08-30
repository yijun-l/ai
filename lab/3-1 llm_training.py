import numpy as np

# Prepare the Dataset
x = np.array([1, 2, 3, 4, 5])
y = 2 * x + 5

# Initialize Parameters
np.random.seed(0)

w = np.random.random()
b = np.random.random()

epochs = 2000
lr = 0.01

# Full Training Loop
for epoch in range(epochs):
    # Forward Propagation (Prediction)
    y_hat = w * x + b

    # Compute Residual Error (Loss: MSE)
    error = y_hat - y

    # Backpropagation (Gradient Computation via Chain Rule)
    # d(MSE)/dw = 2 * mean((y_hat - y) * x)
    # d(MSE)/db = 2 * mean(y_hat - y)
    dw = 2 * np.mean(error * x)
    db = 2 * np.mean(error)

    # Parameter Update
    w = w - dw * lr
    b = b - db * lr

    if epoch % 200 == 0:
        print(f'Epoch [{epoch:4d}/{epochs}] | w: {w:.4f} | b: {b:.4f}')

print("-" * 45)
print(f'Trained Model: Y = {w:.1f} * X + {b:.1f}')