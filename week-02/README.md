# Week 2 — Building a Transformer from Scratch

Following Andrej Karpathy's *"Let's build GPT: from scratch"* and reading the
original *Attention Is All You Need* paper. Goal: implement a decoder-only
transformer in plain PyTorch and understand every line — because the same
architecture (with image/state tokens instead of characters) is what powers the
robot policies later in this ramp (ACT, π0, OpenVLA).

All models are **character-level language models** trained on ~1M characters of
Tiny Shakespeare.

## The progression

Each file adds one idea on top of the last:

| File | What it adds | Val loss |
|------|--------------|---------:|
| `01_bigram.py` | Bigram baseline — each char predicts the next from a lookup table | ~2.50 |
| `02_attention.py` | The self-attention mechanism itself (the math), built up step by step | — |
| `03_attention_lm.py` | A single multi-head self-attention layer + token/positional embeddings | ~2.4 |
| `04_transformer.py` | **The full transformer** — FFN, residuals, LayerNorm, 6 stacked blocks | **1.49** |

## What makes `04` a real transformer

Four additions turn the single attention layer of `03` into a GPT:

1. **FeedForward** — a per-position MLP (4× inner expansion). Attention lets
   tokens *communicate*; the FFN lets each token *compute* on what it gathered.
2. **Block** — (attention + FFN) wrapped with **residual connections**
   (`x = x + sublayer(x)`, a clean gradient highway) and **pre-norm LayerNorm**.
3. **Depth** — 6 stacked blocks, so the network builds increasingly abstract
   features layer by layer.
4. **LayerNorm + Dropout** — normalization that keeps a deep stack trainable and
   regularization that stops it memorizing the training text.

Config: `block_size=128, n_embd=192, n_head=6, n_layer=6, dropout=0.2`,
AdamW @ `3e-4`, 5000 iterations. **2.72M parameters**, trained in a few minutes
on an Apple Silicon GPU (MPS).

## Results

Validation loss fell from **3.72 → 1.49** over 5000 steps:

```
Step    0 | train 3.71 | val 3.72
Step 1000 | train 1.58 | val 1.76
Step 3000 | train 1.30 | val 1.53
Step 4999 | train 1.23 | val 1.49
```

A sample from the trained model (`transformer_sample.txt`) — note the real
character names, the `NAME:` play formatting, and the archaic diction, none of
which were hand-coded:

```
TYBALT:
You loved awhile, who come weep instant, we perceive drop:
where I am receall'd.

...

KING RICHARD II:
Why, good my lord, Camillo,
Sits thy such as dishonourity:
Unmity, I but they'll do advised with their senate;
Even men, father's mine hence; and bear him to cquite
Romeo's a greatful thirst of reversave.
```

Still gibberish word-by-word (a 2.7M char-level model has no real dictionary),
but it has clearly learned the *structure* of a Shakespeare play.

## Reproduce

```bash
python 04_transformer.py                       # full 5000-iter run
ITERS=50 EVAL_INTERVAL=25 python 04_transformer.py   # quick smoke test
```

## What I learned

- **Residual connections are the thing that makes depth work.** Removing them
  (or post-norm instead of pre-norm) and a 6-layer net trains far worse.
- The FFN is where most of the parameters live, and attention is the cheaper
  "routing" layer — a useful intuition for why inference cost scales the way it does.
- Initial loss ≈ `ln(vocab_size)` is a free sanity check that the model and init
  aren't broken before you spend time training.

## → Why this matters for robotics

This is the architecture under every modern Vision-Language-Action model. Swap
character tokens for (image patches + robot state) and the next-token objective
for next-action, and `04_transformer.py` is structurally the backbone of ACT and
π0. Finishing this means those policies won't be black boxes.
