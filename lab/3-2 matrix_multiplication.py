import numpy as np

if __name__ == "__main__":
    # Base Input Matrix X (Shape: 2 houses x 3 basic features)
    X = np.array([
        [10.0, 0.5, 9.0],  # House A: 100m², 5 years old, score 9/10
        [5.0, 2.0, 2.0]  # House B: 50m², 20 years old, score 2/10
    ])

    # Weight Matrix W: 3 basic features -> 2 high-level indicators
    W = np.array([
        [1.0, -0.1],  # Area weights
        [-2.0, -2.0],  # Age weights
        [1.0, 1.0]  # Decoration weights
    ])

    # Perform GEMM: Y = X @ W (Shape: 2x3 @ 3x2 -> 2x2)
    Y = X @ W

    print("GEMM Output Y (Shape: 2x2):")
    print(Y)

    # 1. Calculate vector lengths (L2 Norm) for each row
    norms = np.linalg.norm(X, axis=1, keepdims=True)

    # 2. Normalize Matrix X so each row has length 1.0
    X_norm = X / norms

    print("\nNormalized Matrix X_norm (Shape: 2x3):")
    print(X_norm)

    X_norm_T = X_norm.T  # Equivalent to np.transpose(X_norm)

    # 2. Compute Cosine Similarity Matrix: (2x3) @ (3x2) -> (2x2)
    Similarity = X_norm @ X_norm_T

    print("\nCosine Similarity Matrix (Shape: 2x2):")
    print(Similarity)