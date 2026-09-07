import numpy as np
from pytorch.Linear import Linear
from pytorch.SGD import SGD

if __name__ == "__main__":

    # Prepare the Dataset (y = 1 * X1 + 2 * X2 + 10)
    X = np.array([[1, 2],
                  [2, 5],
                  [3, 4],
                  [2, 5],
                  [2, 6]])

    W_real = np.array([[1],
                       [2]])

    Y = X @ W_real + 10

    # Instantiate model and optimizer
    # W.shape = (2,1)
    model = Linear(in_features=2, out_features=1)
    optimizer = SGD(model, lr=0.02)

    epochs = 2200

    # Full Training Loop
    for epoch in range(epochs):
        # Step 1: Zero out historical gradients
        optimizer.zero_grad()

        # Step 2: Forward pass
        Y_pred = model.forward(X)

        # Step 3: Compute derivative of Loss with respect to y_pred
        # Loss function: MSE = mean((y_pred - y)^2)
        # Derivative: d(MSE) / d(y_pred) = 2 * (y_pred - y)
        dY_pred = 2 * (Y_pred - Y)

        # Step 4: Backward pass (compute dw and db)
        model.backward(dY_pred)

        # Step 5: Parameter update
        optimizer.step()

        if epoch % 200 == 0:
            print(f'Epoch [{epoch:4d}/{epochs}] | w1: {model.W[0,0]:.4f} | w2: {model.W[1,0]:.4f} | b: {model.b:.4f} | loss: { np.mean((Y_pred - Y) ** 2):.4f}')

    print("-" * 45)
    print(f'Trained Model: Y = {model.W[0,0]:.1f} * X1 + {model.W[1,0]:.1f} * X2 + {model.b:.1f}')