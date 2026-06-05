"""
Week 2 Day 2 — Self-Attention from Scratch
Build the core operation of the transformer, one step at a time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# =============================================================
# Setup: a toy "embedded" sequence
# =============================================================
# Imagine these are token embeddings from your bigram (or any embedding).
# We have a batch of 4 sequences, each of length 8, each token is 32-dim.

B, T, C = 4, 8, 32
x = torch.randn(B, T, C)
print(f"Input shape: {x.shape}")  # (4, 8, 32)


# =============================================================
# Version 1: The naive average
# =============================================================
# Goal: each output position is the AVERAGE of all previous positions
# (including itself). No learning yet. Just averaging.

print("\n--- Version 1: Naive average ---")

xbow = torch.zeros((B, T, C))  # "bag of words" — will hold averages
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]   # shape (t+1, C): all tokens up to and including t
        xbow[b, t] = xprev.mean(dim=0)  # average them

print(f"Output shape: {xbow.shape}")  # (4, 8, 32)
print(f"At position 0, xbow == x[..., 0, :]? {torch.allclose(xbow[0, 0], x[0, 0])}")
print(f"At position 1, xbow is average of x[0] and x[1]? "
      f"{torch.allclose(xbow[0, 1], (x[0, 0] + x[0, 1]) / 2)}")


# =============================================================
# Version 2: The same thing, but with matrix multiplication
# =============================================================
# The for-loops in Version 1 are inefficient. We can do the same
# operation with one matrix multiplication.
#
# Key insight: if we have a (T, T) matrix where each row is "uniform
# weights over previous positions," then multiplying it by our (T, C)
# token matrix gives us the weighted averages.

print("\n--- Version 2: Same thing, via matmul ---")

# Build the "averaging" weight matrix.
# Row t should have 1/(t+1) for positions 0..t, and 0 for positions t+1..T-1
wei = torch.tril(torch.ones(T, T))   # lower triangular matrix of 1s
print(f"Weight matrix BEFORE normalization:\n{wei}")
wei = wei / wei.sum(dim=1, keepdim=True)  # normalize each row to sum to 1
print(f"\nWeight matrix AFTER normalization:\n{wei}")

# Apply it: (T, T) @ (B, T, C) -> (B, T, C)
xbow2 = wei @ x   # PyTorch broadcasts wei across the batch dim
print(f"\nxbow2 shape: {xbow2.shape}")
print(f"Same result as Version 1? {torch.allclose(xbow, xbow2)}")


# =============================================================
# Version 3: Using softmax (the form attention actually uses)
# =============================================================
# Instead of dividing by counts, we use softmax to normalize.
# This is the form that generalizes to learned weights.

print("\n--- Version 3: Same thing, via softmax ---")

tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))  # block out future positions
print(f"Weight matrix BEFORE softmax (note the -inf in upper triangle):\n{wei}")
wei = F.softmax(wei, dim=-1)
print(f"\nWeight matrix AFTER softmax:\n{wei}")

xbow3 = wei @ x
print(f"\nSame result? {torch.allclose(xbow, xbow3)}")


# =============================================================
# Version 4: REAL self-attention with learned weights
# =============================================================
# Now we make the weights LEARNABLE based on the data itself.
# Each token produces three vectors via three learned projections:
#   - QUERY: "what am I looking for?"
#   - KEY: "what do I represent?"
#   - VALUE: "what information do I carry?"
#
# Weight between positions i and j = dot(query_i, key_j).
# High dot product = high attention weight.

print("\n--- Version 4: REAL self-attention with Q, K, V ---")

head_size = 16

# Three linear projections — these are the learnable parts
key   = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

# Project the inputs
k = key(x)     # (B, T, head_size)
q = query(x)   # (B, T, head_size)
v = value(x)   # (B, T, head_size)

print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

# Compute attention scores: dot product of each query with each key
# q @ k.transpose(-2, -1): (B, T, head_size) @ (B, head_size, T) = (B, T, T)
wei = q @ k.transpose(-2, -1)
print(f"Attention scores shape: {wei.shape}")  # (B, T, T)

# Scale by sqrt(head_size) for numerical stability (we'll discuss why)
wei = wei * head_size ** -0.5

# Causal mask: position i can only attend to positions 0..i
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))

# Softmax to convert to probabilities
wei = F.softmax(wei, dim=-1)

# Apply weights to values
out = wei @ v   # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
print(f"Output shape: {out.shape}")  # (B, T, head_size)

# Inspect one attention pattern
print(f"\nAttention weights for sequence 0:\n{wei[0]}")
print(f"\n(Notice: each row sums to 1, lower-triangular — each position only attends to itself and earlier positions)")







# =============================================================
# Step 1: Wrap single-head attention into a reusable class
# =============================================================

class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Register the causal mask as a buffer (not a learnable parameter)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x shape: (B, T, n_embd)
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        # Apply causal mask (only attend to past)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        # Apply weights to values
        out = wei @ v  # (B, T, head_size)
        return out
    

print("\n--- Single head wrapped in a class ---")
head = Head(head_size=16, n_embd=C, block_size=T)
out = head(x)
print(f"Output shape: {out.shape}")  # (4, 8, 16)



# =============================================================
# Step 2: Multi-head attention — naive parallel implementation
# =============================================================

class MultiHeadAttention(nn.Module):
    """Run multiple single-head attention modules in parallel."""

    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        # Create num_heads separate single-head modules
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size) for _ in range(num_heads)
        ])
        # Final projection layer (after concatenation)
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        # Run each head, get (B, T, head_size) tensors
        # Concatenate along the last axis: (B, T, num_heads * head_size)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Project back to n_embd
        out = self.proj(out)
        return out
    

print("\n--- Multi-head attention ---")
mha = MultiHeadAttention(num_heads=4, head_size=8, n_embd=C, block_size=T)
out = mha(x)
print(f"Output shape: {out.shape}")  # (4, 8, 32)