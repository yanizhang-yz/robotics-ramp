"""
# Week 2 Day 1 - Bigram Language Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

torch.manual_seed(1337)

# ============================================
# 1. Load and encode the data
# ============================================

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
filename = "shakespeare.txt"

if not os.path.exists(filename):
    print(f"Downloading shakespeare...")
    urllib.request.urlretrieve(url, filename)

with open(filename, "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

char_to_idx = {ch:i for i, ch in enumerate(chars)}
idx_to_char = {i:ch for i, ch in enumerate(chars)}

data = torch.tensor([char_to_idx[ch] for ch in text], dtype = torch.long)

# Split into train (first 90%) and validation (last 10%)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

# ============================================
# 2. Batching
# ============================================

batch_size = 32
block_size = 8

def get_batch(split):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

xb, yb = get_batch("train")
print(f"xb shape: {xb.shape}")  # (32, 8)
print(f"yb shape: {yb.shape}")  # (32, 8)

# ============================================
# 3. Bigram model
# ============================================

class BigramLanguageModel(nn.Module):
    """
    The whole model is a lookup table.
    Input: character index. Output: vocab_size logits over the next character.
    No context, no history.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

    # ============================================
    # 4. Training
    # ============================================  

model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

num_iterations = 3000
eval_interval = 300

@torch.no_grad()
def estimate_loss():
    model.eval()
    loses = {}
    for split in ["train", "val"]:
        batch_losses = []
        for _ in range(20):
            x, y = get_batch(split)
            _, loss = model(x, y)
            batch_losses.append(loss.item())
        loses[split] = sum(batch_losses) / len(batch_losses)
    model.train()
    return loses

print("Starting training...")
for iter in range(num_iterations):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()        
    if iter % eval_interval == 0 or iter == num_iterations - 1:
        losses = estimate_loss()
        print(f"Step {iter}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")


print("\nTraining done. \n")

# ============================================
# 5. Generate some text
# ============================================          

start_token = char_to_idx["\n"]
context = torch.tensor([[start_token]], dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)[0].tolist()
print("Generated text:")
print(''.join(idx_to_char[i] for i in generated))