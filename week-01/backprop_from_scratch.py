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


# ============================================
# Loss: softmax + cross-entropy
# ============================================
def softmax_cross_entropy(logits, y):
    """
    Compute softmax probabilities and cross-entropy loss.
    Returns: average loss (scalar), probabilities (N, num_classes)
    """
    # Numerical stability: subtract the max of each row before exponentiating
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Cross-entropy: -log(prob assigned to correct class)
    N = logits.shape[0]
    correct_log_probs = -np.log(probs[np.arange(N), y] + 1e-12)  # tiny epsilon to avoid log(0)
    loss = np.mean(correct_log_probs)

    return loss, probs

# Test the loss
loss, probs = softmax_cross_entropy(logits, y)
print(f"\nInitial loss: {loss:.4f}")
print(f"Expected if predictions were random: {np.log(output_dim):.4f}")

# ============================================
# Backward pass — compute gradients by hand
# ============================================
def backward(probs, y, cache, W2):
    """
    Compute gradients of the loss with respect to W1, b1, W2, b2.
    """
    X = cache["X"]
    z1 = cache["z1"]
    h1 = cache["h1"]
    N = X.shape[0]

    # ------ Step 1: gradient at the logits ------
    # dL/dz2 = probs - one_hot(y), then average across batch
    dz2 = probs.copy()
    dz2[np.arange(N), y] -= 1
    dz2 /= N  # average across batch

    # ------ Step 2: gradients for layer 2 weights ------
    # dL/dW2 = h1.T @ dz2
    # dL/db2 = sum of dz2 across batch
    dW2 = h1.T @ dz2
    db2 = np.sum(dz2, axis=0)

    # ------ Step 3: propagate gradient back to h1 ------
    # dL/dh1 = dz2 @ W2.T
    dh1 = dz2 @ W2.T

    # ------ Step 4: gradient through ReLU ------
    # dL/dz1 = dh1 where z1 > 0, else 0
    dz1 = dh1 * (z1 > 0)

    # ------ Step 5: gradients for layer 1 weights ------
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return dW1, db1, dW2, db2

# Test the backward pass
dW1, db1, dW2, db2 = backward(probs, y, cache, W2)
print(f"\ndW1 shape: {dW1.shape}, expected {W1.shape}")
print(f"db1 shape: {db1.shape}, expected {b1.shape}")
print(f"dW2 shape: {dW2.shape}, expected {W2.shape}")
print(f"db2 shape: {db2.shape}, expected {b2.shape}")


# ============================================
# Training loop
# ============================================
# Re-initialize so we train fresh
np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
b2 = np.zeros(output_dim)

learning_rate = 0.1
num_iterations = 2000

losses = []
for i in range(num_iterations):
    # Forward
    logits, cache = forward(X, W1, b1, W2, b2)
    loss, probs = softmax_cross_entropy(logits, y)

    # Backward
    dW1, db1, dW2, db2 = backward(probs, y, cache, W2)

    # Update weights — gradient descent
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    losses.append(loss)
    if i % 200 == 0:
        # Compute accuracy
        predictions = np.argmax(logits, axis=1)
        accuracy = np.mean(predictions == y)
        print(f"Iter {i:4d} | loss: {loss:.4f} | accuracy: {accuracy*100:.1f}%")

# Final evaluation
final_logits, _ = forward(X, W1, b1, W2, b2)
final_predictions = np.argmax(final_logits, axis=1)
final_accuracy = np.mean(final_predictions == y)
print(f"\nFinal accuracy: {final_accuracy*100:.1f}%")


# ============================================
# Visualize the decision boundary
# ============================================
# Create a fine grid covering the input space
grid_x, grid_y = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 200),
    np.linspace(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, 200),
)
grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

# Predict on the grid
grid_logits, _ = forward(grid_points, W1, b1, W2, b2)
grid_predictions = np.argmax(grid_logits, axis=1).reshape(grid_x.shape)

# Plot
plt.figure(figsize=(7, 7))
plt.contourf(grid_x, grid_y, grid_predictions, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='black')
plt.title(f"Learned decision boundary (final accuracy: {final_accuracy*100:.1f}%)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.savefig("decision_boundary.png", dpi=100, bbox_inches='tight')
plt.show()
print("Saved decision_boundary.png")