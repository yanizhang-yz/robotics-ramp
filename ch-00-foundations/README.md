# Chapter 0 — Foundations

PyTorch and the fundamentals, from scratch, before the transformer. The groundwork the
[ROADMAP](../ROADMAP.md) learning track builds on.

| File | What |
|------|------|
| `01_mnist_feedforward.ipynb` | Feedforward net on MNIST — the "hello world" of training loops |
| `02_cifar10_cnn.ipynb` | A CNN on CIFAR-10 — convolutions, pooling, augmentation |
| `03_rnn_char.ipynb` | Character-level RNN on Shakespeare — sequence modeling before attention |
| `backprop_from_scratch.py` | Backprop by hand in NumPy on a 2-layer net (spiral data) |
| `04_autograd.py` | A tiny autograd engine — what `loss.backward()` actually does |

**Why it's Chapter 0:** the ROADMAP's learning queue starts at the transformer (Ch 1).
This is the prerequisite layer that came first — kept because the from-scratch
backprop/autograd work is the foundation everything else stands on.
