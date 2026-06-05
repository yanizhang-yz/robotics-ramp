"""
Week 2 Day 5 — A Full Transformer (decoder-only GPT)

The final step of the Karpathy "Let's build GPT" build. This takes the single
self-attention layer from 03_attention_lm.py and turns it into a real
transformer by adding the four pieces a working GPT needs:

  1. FeedForward  — a per-position MLP. Attention lets tokens *communicate*
                    (gather info from other positions); the FFN lets each token
                    *think* on what it gathered. Communication then computation.

  2. Block        — (attention + FFN) wrapped with RESIDUAL CONNECTIONS and
                    pre-LayerNorm. This is the repeatable unit of a transformer.

  3. Depth        — we stack N_LAYER blocks so the network can build up
                    increasingly abstract features layer by layer.

  4. LayerNorm + Dropout — normalization that keeps a deep stack trainable, and
                    regularization that stops it memorizing the training text.

Everything here is the same architecture behind every modern LLM — and, with
image/state tokens swapped in for characters, behind the VLA robot policies
(ACT, pi0, OpenVLA) we'll train later in the ramp. That's why this is worth
finishing properly before touching robots again.

Run it:
    python 04_transformer.py
Quick smoke test (tiny run just to prove it trains end-to-end):
    ITERS=50 EVAL_INTERVAL=25 python 04_transformer.py
"""

import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================
# Hyperparameters
# ============================================
# Defaults below are tuned to train in a few minutes on Apple Silicon (MPS)
# while still producing recognizably Shakespeare-like text.
#
# Karpathy's "classic" config for noticeably better samples (slower on MPS):
#     N_EMBD=384, NUM_HEADS=6, N_LAYER=6, BLOCK_SIZE=256, BATCH_SIZE=64  (~10M params)
BATCH_SIZE = 64
BLOCK_SIZE = 128          # context length: how many chars of history the model sees
N_EMBD = 192              # embedding / residual-stream width
NUM_HEADS = 6             # head_size = N_EMBD // NUM_HEADS = 32
N_LAYER = 6               # number of stacked transformer blocks (the new "depth")
DROPOUT = 0.2             # dropout probability (new — regularization)
LEARNING_RATE = 3e-4      # lower LR than the single-layer model — deeper nets are touchier
NUM_ITERATIONS = int(os.environ.get("ITERS", 5000))
EVAL_INTERVAL = int(os.environ.get("EVAL_INTERVAL", 500))
EVAL_BATCHES = 20         # batches used to estimate loss at each eval step

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
    """One head of self-attention.

    Unchanged from 03 except for one addition: DROPOUT on the attention
    weights. Randomly zeroing some attention connections during training
    stops any single token->token edge from becoming load-bearing.
    """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Causal mask: a buffer (moves with the model, but isn't a learned parameter)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                          # (B, T, head_size)
        q = self.query(x)                                        # (B, T, head_size)

        # Scaled dot-product attention scores
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)    # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # causal mask
        wei = F.softmax(wei, dim=-1)                             # (B, T, T)
        wei = self.dropout(wei)                                  # NEW: regularize the attention

        v = self.value(x)                                        # (B, T, head_size)
        out = wei @ v                                            # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated then projected.

    Addition over 03: a DROPOUT after the output projection. The projection
    itself (self.proj) is what lets attention write back into the residual
    stream — keep an eye on it, residual connections depend on it.
    """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(num_heads * head_size, n_embd)    # back to residual width
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)     # (B, T, num_heads*head_size)
        out = self.dropout(self.proj(out))                      # (B, T, n_embd)
        return out


class FeedForward(nn.Module):
    """Per-position MLP — the 'computation' half of a transformer block. (NEW)

    Runs independently on every position. The 4x inner expansion (n_embd ->
    4*n_embd -> n_embd) is the standard transformer ratio: it gives the token
    room to do nonlinear work on the information attention just gathered.
    """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),   # projection back into the residual stream
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """A transformer block: communication (attention) THEN computation (FFN). (NEW)

    Two ideas that make deep transformers actually train, both visible here:

      * RESIDUAL CONNECTIONS — `x = x + sublayer(x)`. The input flows around
        each sub-layer untouched, so gradients have a clean highway back to
        the start. Sub-layers only learn a *delta* on top of the input.

      * PRE-NORM LAYERNORM — we LayerNorm the *input* to each sub-layer
        (ln1, ln2) rather than the output. Pre-norm is the modern default;
        it's what keeps that gradient highway numerically stable with depth.
    """

    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))      # residual around attention
        x = x + self.ffwd(self.ln2(x))    # residual around feed-forward
        return x


class GPTLanguageModel(nn.Module):
    """Decoder-only transformer. The single attention layer from 03 is now
    one repeatable Block, stacked N_LAYER deep, with a final LayerNorm."""

    def __init__(self, vocab_size, n_embd, num_heads, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        # NEW: a stack of blocks instead of a single attention layer
        self.blocks = nn.Sequential(*[
            Block(n_embd, num_heads, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)            # NEW: final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)             # Karpathy-style scaled init

    def _init_weights(self, module):
        """Small-std init keeps early-training logits calm — initial loss
        should land near ln(vocab_size)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)                                   # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = tok_emb + pos_emb                                                 # (B, T, n_embd)

        x = self.blocks(x)                                                    # (B, T, n_embd)
        x = self.ln_f(x)                                                      # (B, T, n_embd)
        logits = self.lm_head(x)                                             # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Autoregressively sample max_new_tokens new characters."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]   # can't attend past block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]              # focus on the last step (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================
# 4. Training utilities
# ============================================
@torch.no_grad()
def estimate_loss(model):
    """Average loss over EVAL_BATCHES batches from each split (less noisy than one)."""
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
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        num_heads=NUM_HEADS,
        n_layer=N_LAYER,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  ({n_params / 1e6:.2f}M)")

    # Sanity check: one forward pass — verify shapes and that initial loss ~ ln(vocab)
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    expected = torch.log(torch.tensor(float(vocab_size))).item()
    print(f"Sanity — logits shape: {tuple(logits.shape)} (expected ({BATCH_SIZE}, {BLOCK_SIZE}, {vocab_size}))")
    print(f"Sanity — initial loss: {loss.item():.4f} (expected ~{expected:.2f})\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Training...")
    train(model, optimizer)
    print("Training done.\n")

    # Generate a sample, print it, and save it as a deliverable for the README
    start = torch.tensor([[char_to_idx["\n"]]], dtype=torch.long, device=DEVICE)
    generated = model.generate(start, max_new_tokens=1000)[0].tolist()
    sample = "".join(idx_to_char[i] for i in generated)

    print("Generated sample (first 500 chars):")
    print(sample[:500])

    out_path = "transformer_sample.txt"
    with open(out_path, "w") as f:
        f.write(sample)
    print(f"\nFull 1000-char sample saved to {out_path}")


if __name__ == "__main__":
    main()
