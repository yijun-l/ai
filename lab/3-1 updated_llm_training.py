import numpy as np
from pytorch.Linear import Linear
from pytorch.SGD import SGD

if __name__ == "__main__":

    # Prepare the Dataset
    x = np.array([1, 2, 3, 4, 5])
    y = 2 * x + 5

    # Instantiate model and optimizer
    model = Linear()
    optimizer = SGD(model)

    epochs = 2000

    # Full Training Loop
    for epoch in range(epochs):
        # Step 1: Zero out historical gradients
        optimizer.zero_grad()

        # Step 2: Forward pass
        y_pred = model.forward(x)

        # Step 3: Compute derivative of Loss with respect to y_pred
        # Loss function: MSE = mean((y_pred - y)^2)
        # Derivative: d(MSE) / d(y_pred) = 2 * (y_pred - y)
        dy_pred = 2 * (y_pred - y)

        # Step 4: Backward pass (compute dw and db)
        model.backward(dy_pred)

        # Step 5: Parameter update
        optimizer.step()

        if epoch % 200 == 0:
            print(f'Epoch [{epoch:4d}/{epochs}] | w: {model.w:.4f} | b: {model.b:.4f}')

    print("-" * 45)
    print(f'Trained Model: Y = {model.w:.1f} * X + {model.b:.1f}')