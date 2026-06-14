# Robotics Ramp

My transition from senior software engineer (8 years) into AI-for-robotics — learned
**depth-first** and built in public. Started **April 29, 2026**.

This repo is the *"understand the field from first principles"* half: **the plan and the
study chapters.** The hands-on hardware work — milestones **M0–M4** — lives in the
companion **`lerobot-experiments`** repo.

## The plan

**[ROADMAP.md](ROADMAP.md) is the single source of truth.** Two tracks:

- **Learning** — the chapters below, pulled depth-first, no deadlines.
- **Shipping** — milestones M0–M4 on a rough clock, executed in `lerobot-experiments`.

## Chapters

| # | Chapter | What | Status |
|---|---------|------|--------|
| 0 | [Foundations](ch-00-foundations/) | PyTorch from scratch: MNIST, CNN, RNN, backprop, autograd | ✅ |
| 1 | [Transformer](ch-01-transformer/) | Decoder-only transformer from scratch (val loss 1.49) | ✅ |
| 2 | [RL core](ch-02-rl/) | CS224R: value iteration, Q-learning, REINFORCE | 🟡 |
| 3 | [Bridge to robotics](ch-03-bridge-to-robotics/) | Behavior cloning, Diffusion Policy, ACT | ⬜ |
| 4 | [VLAs](ch-04-vlas/) | Vision-Language-Action models; VLA vs world-models | ⬜ |
| 5 | [Policy architectures](ch-05-policy-architectures/) | How the pieces fit in the LeRobot stack | ⬜ |

## How I work

Build first, *then* diff against the reference implementation — that's where the
understanding lands. Code committed in public even when ugly; failures documented
alongside the wins.

## Stack

Python 3.12 · PyTorch (MPS / Apple Silicon) · Hugging Face LeRobot · MuJoCo · Weights & Biases

---

*Planning archive: the original week-by-week [CHECKLIST](archive/CHECKLIST.md) (superseded by
the ROADMAP) and the [capstone path options](CAPSTONE_PATHS.md).*
