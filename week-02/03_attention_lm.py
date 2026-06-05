"""
Week 2 Day 4 — Self-Attention Language Model

A character-level language model with multi-head self-attention.
Builds on top of the bigram model from 01_bigram.py.

Key additions over the bigram:
- Token embeddings now have a separate dimension (n_embd) from vocab size
- Positional embeddings let the model know where each token sits
- Multi-head self-attention mixes information across positions
- A final linear projection maps n_embd -> vocab_size
"""

import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================
# Hyperparameters (all in one place for easy tweaking)
# ============================================
BATCH_SIZE = 32
BLOCK_SIZE = 8           # context length
N_EMBD = 32              # embedding dimension
NUM_HEADS = 4            # multi-head attention
LEARNING_RATE = 1e-3     # lower than bigram's 1e-2 — attention is more sensitive
NUM_ITERATIONS = 5000
EVAL_INTERVAL = 500
EVAL_BATCHES = 20        # batches used to estimate loss at each eval step

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(1337)

print(f"Using device: {DEVICE}")


# ============================================
# 1. Load and encode the data
# ============================================
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
FILENAME = "shakespeare.txt"

if not os.path.exists(FILENAME):
    print("Downloading Shakespeare...")
    urllib.request.urlretrieve(URL, FILENAME)

with open(FILENAME, "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Vocab size: {vocab_size}")
print(f"Train size: {len(train_data):,}, Val size: {len(val_data):,}")


# ============================================
# 2. Batching
# ============================================
def get_batch(split):
    """Sample a batch of (input, target) sequences."""
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([source[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ============================================
# 3. Model components
# ============================================
class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Causal mask: stored as a buffer (moves with the model but isn't learned)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                          # (B, T, head_size)
        q = self.query(x)                                        # (B, T, head_size)
        v = self.value(x)                                        # (B, T, head_size)

        # Attention scores, scaled
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)    # (B, T, T)
        # Causal mask — only attend to past positions
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)                             # (B, T, T)

        out = wei @ v                                            # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention running in parallel."""

    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)      # (B, T, num_heads * head_size)
        out = self.proj(out)                                     # (B, T, n_embd)
        return out


class AttentionLanguageModel(nn.Module):
    """Character-level language model with one multi-head self-attention layer."""

    def __init__(self, vocab_size, n_embd, num_heads, block_size):
        super().__init__()
        self.block_size = block_size

        # Token embedding: each character -> n_embd vector
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # Positional embedding: each position 0..block_size-1 -> n_embd vector
        self.position_embedding = nn.Embedding(block_size, n_embd)
        # Multi-head self-attention
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, n_embd, block_size)
        # Final projection from n_embd back to vocab logits
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)                                          # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))        # (T, n_embd)
        x = tok_emb + pos_emb                                                        # (B, T, n_embd)

        x = self.sa(x)                                                               # (B, T, n_embd)
        logits = self.lm_head(x)                                                     # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens (model can't see further back)
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                          # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)            # (B, T+1)
        return idx


# ============================================
# 4. Training utilities
# ============================================
@torch.no_grad()
def estimate_loss(model):
    """Compute average loss over EVAL_BATCHES batches from train and val splits."""
    model.eval()
    out = {}
    for split in ["train", "val"]:
        batch_losses = []
        for _ in range(EVAL_BATCHES):
            x, y = get_batch(split)
            _, loss = model(x, y)
            batch_losses.append(loss.item())
        out[split] = sum(batch_losses) / len(batch_losses)
    model.train()
    return out


def train(model, optimizer):
    """Main training loop."""
    for step in range(NUM_ITERATIONS):
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == NUM_ITERATIONS - 1:
            losses = estimate_loss(model)
            print(f"Step {step:4d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")


# ============================================
# 5. Run it
# ============================================
def main():
    model = AttentionLanguageModel(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        num_heads=NUM_HEADS,
        block_size=BLOCK_SIZE,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Sanity check: run one forward pass, verify shapes and initial loss
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    print(f"Sanity check — logits shape: {logits.shape} (expected ({BATCH_SIZE}, {BLOCK_SIZE}, {vocab_size}))")
    print(f"Sanity check — initial loss: {loss.item():.4f} (expected ~{torch.log(torch.tensor(float(vocab_size))).item():.2f})")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Training...")
    train(model, optimizer)
    print("Training done.\n")

    # Generate
    start_token = char_to_idx["\n"]
    context = torch.tensor([[start_token]], dtype=torch.long, device=DEVICE)
    generated = model.generate(context, max_new_tokens=500)[0].tolist()
    print("Generated text:")
    print("".join(idx_to_char[i] for i in generated))


if __name__ == "__main__":
    main()