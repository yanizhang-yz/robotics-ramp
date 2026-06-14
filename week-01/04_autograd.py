"""
Tensors & Autograd
------------------
Goal: rebuild your spiral-dataset in PyTorch, but let AUTOGRAD compute the gradients
instead of your hand-written Numpy backward. Then verify autograd's gradients match a
numerical (finite-difference) gradient - the moment where you see that autograd is
doing exactly what you expect is a magical moment.
"""

import torch
torch.manual_seed(0)

# ----------------------------------------------
# 1. Spiral dataset
# ----------------------------------------------
def make_spiral(points_per_class=100, num_classes=3):
    N, K = points_per_class, num_classes
    X = torch.zeros(N*K, 2)
    y = torch.zeros(N*K, dtype=torch.long)
    for k in range(K):
        ix = slice(N*k, N*(k+1))
        r = torch.linspace(0.0, 1, N)  # radius
        t = torch.linspace(k*4, (k+1)*4, N) + torch.randn(N) * 0.2  # theta
        X[ix] = torch.stack([r * torch.sin(t), r * torch.cos(t)], dim=1)
        y[ix] = k
    return X, y

X, y = make_spiral(points_per_class=100, num_classes=3)
print(f"X shape: {X.shape}")  # (300, 2) — 300 points, 2 features each
print(f"y shape: {y.shape}")  # (300,) — 300 labels

# ----------------------------------------------
# 2. Parameters as LEAF tensors that require
#    grad A 2-layer MLP: 2 -> H -> num_classes
# ----------------------------------------------
H = 100  # hidden layer size
W1 = torch.randn(2, H, requires_grad=True)  # (2, 100)
b1 = torch.zeros(H, requires_grad=True)      # (100,)
W2 = torch.randn(H, 3, requires_grad=True)   # (100, 3)
b2 = torch.zeros(3, requires_grad=True)      # (3,)
params = [W1, b1, W2, b2]
print(f"W1 shape: {W1.shape}, requires_grad: {W1.requires_grad}")
print(f"b1 shape: {b1.shape}, requires_grad: {b1.requires_grad}")
print(f"W2 shape: {W2.shape}, requires_grad: {W2.requires_grad}")
print(f"b2 shape: {b2.shape}, requires_grad: {b2.shape}, requires_grad: {b2.requires_grad}")

# ----------------------------------------------
# 3. Forward pass: compute logits and loss
# ----------------------------------------------
def forward(X):
    z1 = X @ W1 + b1  # (300, H)
    h1 = torch.relu(z1)  # (300, H)
    logits = h1 @ W2 + b2  # (300, 3)
    return logits

# ----------------------------------------------
# 4. Loss: cross-entropy
# ----------------------------------------------
def cross_entropy(logits, y):
    N, C = logits.shape  # logits: (N, C), y: (N,)
    log_probs = torch.log_softmax(logits, dim=1)  # (N, C)
    loss = -log_probs[torch.arange(N), y].mean()  # average negative log-likelihood
    return loss

# ----------------------------------------------------------------------
# 5. Numerical gradient check  (GIVEN — use it, don't rewrite it)
#    Central finite difference: (L(w+eps) - L(w-eps)) / (2*eps)
# ----------------------------------------------------------------------
@torch.no_grad()
def numerical_grad(param, eps=1e-4):
    grad = torch.zeros_like(param)
    flat, gflat = param.view(-1), grad.view(-1)
    for i in range(flat.numel()):
        orig = flat[i].item()
        flat[i] = orig + eps
        loss_plus = cross_entropy(forward(X), y).item()
        flat[i] = orig - eps
        loss_minus = cross_entropy(forward(X), y).item()
        flat[i] = orig
        gflat[i] = (loss_plus - loss_minus) / (2 * eps)
    return grad
 
# ----------------------------------------------------------------------
# 6. One backward pass + the gradient check  (THE payoff)
# ----------------------------------------------------------------------
loss = cross_entropy(forward(X), y)
loss.backward()
# print W1.grad  (this came from autograd)
print(f"W1.grad shape: {W1.grad.shape}, requires_grad: {W1.grad.requires_grad}")

# compute numerical_grad(W1) and compare with torch.allclose(..., atol=1e-2)
# Also print the max absolute difference. They should match. That match is
# the whole point: autograd == the chain rule you did by hand.
num_grad_W1 = numerical_grad(W1)
print(f"Numerical grad W1 shape: {num_grad_W1.shape}, requires_grad: {num_grad_W1.requires_grad}")
print(f"Max absolute difference in W1 grad: {torch.max(torch.abs(W1.grad - num_grad_W1)):.6f}")
print(f"Autograd grad W1 close to numerical grad W1: {torch.allclose(W1.grad, num_grad_W1, atol=1e-2)}")
 
# ----------------------------------------------------------------------
# 7. Train with manual SGD  (touches zero_grad)
# ----------------------------------------------------------------------
lr = 1.0
for epoch in range(200):
    logits = forward(X)
    loss = cross_entropy(logits, y)
 
    # zero the gradient of each param BEFORE backward
    # (tip: `p.grad = None` works even on the very first iteration)
    for p in params: p.grad = None

    loss.backward()
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad
 
    if epoch % 20 == 0:
        with torch.no_grad():
            # TODO: accuracy = mean( argmax(logits) == y )
            accuracy = (logits.argmax(dim=1) == y).float().mean().item()
            print(epoch, loss, accuracy)
            pass
 
# ----------------------------------------------------------------------
# 8. EXPERIMENT for question 2
#    Comment out your zero-grad line in the loop above, run again, and watch
#    what happens to the loss. Write down what you see and why.
# ----------------------------------------------------------------------
 