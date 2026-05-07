import numpy as np
import matplotlib.pyplot as plt

# Reproducibility — same random numbers every run
np.random.seed(42)

def make_spiral_data(num_per_class=100, num_classes=2, noise=0.2):
    """Generate a 2D spiral dataset. Hard for linear classifiers, easy for NNs."""
    X = np.zeros((num_per_class * num_classes, 2))
    y = np.zeros(num_per_class * num_classes, dtype=int)
    for class_idx in range(num_classes):
        ix = range(num_per_class * class_idx, num_per_class * (class_idx + 1))
        r = np.linspace(0.0, 1, num_per_class)  # radius
        t = (np.linspace(class_idx * 4, (class_idx + 1) * 4, num_per_class)
             + np.random.randn(num_per_class) * noise)  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = class_idx
    return X, y

X, y = make_spiral_data(num_per_class=100, num_classes=2)
print(f"X shape: {X.shape}")  # (200, 2) — 200 points, 2 features each
print(f"y shape: {y.shape}")  # (200,) — 200 labels

# Visualize
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='black')
plt.title("Spiral dataset")
plt.xlabel("x1")
plt.ylabel("x2")
plt.savefig("spiral_data.png", dpi=100, bbox_inches='tight')
plt.show()
print("Saved spiral_data.png")


# =================================
# Network arhchitecture
# =================================
input_dim = 2 # 2D points
hidden_dim = 16 # 16 neurons in the hidden layer
output_dim = 2 # 2 classes

# Initialize weights with small random values (Xavier initialization)
W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
b2 = np.zeros(output_dim)

print(f"\nW1 shape: {W1.shape}")  # (2, 16)
print(f"b1 shape: {b1.shape}")  # (16,)
print(f"W2 shape: {W2.shape}")  # (16, 2)
print(f"b2 shape: {b2.shape}")  # (2,)

# =================================
# Forward pass
# =================================
def forward(X, W1, b1, W2, b2):
    """Compute logits for input X. Also return intermediate values for backprop."""
    # Layer 1: linear -> ReLU
    z1 = X @ W1 + b1 # (N, hidden_dim)
    h1 = np.maximum(0, z1) # ReLU: zero out negatives

    # Layer 2: linear (no activation; loss handles softmax)
    z2 = h1 @ W2 + b2 # (N, output_dim)

    # Cache intermediate values - we'll need them in backprop
    cache = {"X": X, "z1": z1, "h1": h1, "z2": z2}
    return z2, cache

# Test forward pass
logits, cache = forward(X, W1, b1, W2, b2)
print(f"\nLogits shape: {logits.shape}")  # (200, 2) - one logit per class for each point
print(f"First 3 rows of logits:\n{logits[:3]}")