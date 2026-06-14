# Robotics Roadmap — Learning + Shipping

A plan built around how I learn — depth-first, curiosity-led, ~4–5 focused hours on
weekdays — now with the home-robotics build spine wired in. The point is momentum and
direction, not daily quotas. Going deep is the goal, not the thing that makes me "behind."

---

## How this works — two tracks

- **Learning** has no dates. A queue I pull from, going as deep as curiosity takes me.
  Nothing on it can be "late." Detail for each item lives in its chapter.
- **Shipping** has a small spine of milestones (M0–M4) in rough sequence. These are the
  only things on a clock, and they exist so curiosity doesn't quietly turn into months of
  drift. Each one ships a **repo subdir + writeup + rollout video + honest success-rate
  metrics** — the shipping spine doubles as my build-in-public pipeline.

Missing a milestone target means I **re-estimate**, not that I failed. Revising the plan is
what the plan is for.

## Daily rhythm

- Sit down for the focused hours I have. Pull the next thing off the queue, or push the
  current milestone. Go deep. Stop when the hours are up. There is no daily checklist.
- **Rabbit-hole check:** when I stop to understand something, notice whether it deepens
  what I'm building now or it's a side quest. Both are fine — I just want a tangent to be a
  choice, not a current that drags me off course.
- **Protected baseline:** garden most days, evening reading, exercise. These keep the
  focused hours sharp — inputs to the work, not competitors. The garden is the canary: many
  days skipped *and* feeling duller rather than sharper = pull back, don't push harder.

---

## Repo layout

Two repos. Do **not** spin up a new repo per experiment — it duplicates infra and hides the
reuse, which is the opposite of the engineering signal I want.

| Repo | Role | Holds |
|---|---|---|
| **`robotics-ramp`** (this repo) | "I understand the field from first principles" | All chapters, skeleton exercises, this `ROADMAP.md` |
| **`robotics-experiments`** | "I can ship on real hardware" | M0–M4, each as a subdirectory, with shared infra (data loading, eval harness, calibration utils) at the repo root and reused across milestones |

**Rule:** one **subdirectory** per experiment, not one repo. Promote a single experiment to
its own standalone repo *only* when it earns a spotlight — polished enough to be a pinned
showcase with its own front door (e.g. the t-shirt-fold if it comes out clean). That's the
exception, driven by "this deserves its own door," never the default.

---

## Track 1 — Learning queue (no dates, pull from the top)

Build first, *then* diff against the reference implementation — that's where the
understanding lands. Each item's full scope + coding builds + exercises live in its chapter.

1. **Finish the transformer** → `Ch 1` — self-attention (Q/K/V), multi-head, full
   forward/backward, train a working model end to end.
2. **RL core via CS224R** (Finn / Hausman) → `Ch 2` — policy gradients, value methods,
   actor-critic, worked question-first.
3. **Bridge to robotics** → `Ch 3` + `Ch 4` — behavior cloning, diffusion policies, ACT,
   then VLAs (and the VLA-vs-world-model distinction) in real depth.
4. **Policy architectures + the LeRobot stack** → `Ch 5` — how the pieces fit for real
   deployment.

> **Reconcile:** I rebuilt this queue at the header level pointing into the chapters. If my
> current `PLAN.md` has more detailed build/exercise notes per item, paste them back under
> the matching chapter so nothing is lost.

---

## Track 2 — Shipping milestones (the home-robotics spine)

The chapters are the reference material; these milestones are where I use them on hardware.
All of them live in **`robotics-experiments`**.

| Milestone | What ships | Chapter | Repo path |
|---|---|---|---|
| **M0 — Bring-up** (Wk 0–1) | Assemble, install LeRobot, calibrate, teleop (leader→follower), both cameras, record + replay one episode. No ML — the first win, de-risks everything. | `Ch 5` | `experiments/m0-bringup` |
| **M1 — First autonomous skill** (Wk 2–4) | Single-object pick-and-place into a bin. ~50 teleop demos → train ACT → deploy → eval over 20 trials → video. Proves the whole loop end to end. | `Ch 3` | `experiments/m1-pick-place` |
| **M2 — Sorting** (Wk 4–8) → *"sorting plates"* | Sort 2–3 object types into bins. Second camera, harder eval (novel positions, lighting). Train **ACT vs Diffusion Policy vs a SmolVLA fine-tune** on one dataset — the head-to-head is its own writeup. | `Ch 3` (+ `Ch 4` for SmolVLA) | `experiments/m2-sorting` |
| **M3 — Deformable stretch** (Wk 8–16) → *"folding clothes," scoped* | Flat t-shirt, one or two folds, ~100+ demos, LeRobot's t-shirt experiment as baseline. Expect brittleness; document exactly where it breaks. | `Ch 4` | `experiments/m3-tshirt-fold` |
| **M4 — Language-conditioned** (Wk 16–24, optional) | Fine-tune the current best open VLA so "put the cup away" vs "fold the shirt" routes to the right skill. Bridge to the orchestration layer and the foundation-model-company story. | `Ch 4` | `experiments/m4-vla` |

**Immediate next move:** M0 if teleop isn't live yet; otherwise start recording M1 demos
today. The bottleneck is data quality, not the model — budget accordingly.

---

## Scope reality — the folding goal (June 2026)

"Sorting plates" and "folding clothes" are opposite ends of the difficulty spectrum, and
keeping that honest is itself a maturity signal to the Pi / Figure / 1X-type companies.

- **Rigid sort / pick-place** is a solved-class *demo* problem on a cheap arm. M1–M2 are
  squarely reachable.
- **General laundry folding is at/beyond the frontier.** At CES 2026, LG's two-armed CLOiD
  — trained on tens of thousands of hours — folded towels into crumpled heaps, needed a
  human to straighten each one, and gave up mid-fold. Cloth has an effectively infinite
  config space, self-occludes, and changes shape every move; most working demos are
  *bimanual*, and I have one arm. So M3 is scoped to a **flat t-shirt fold on a table** — a
  real, reachable slice of the frontier problem, with its limits stated.

## Where I'm strong / the full stack

Bottom → top, with where I sit:

1. Hardware / mechatronics — *newest*
2. Firmware / low-level control — *newest*
3. Middleware / robot software (teleop, recording) — **strong (it's software)**
4. Perception (pixels→state; deformables are the wall) — *gap, esp. cloth*
5. Policy / learning (ACT, Diffusion, VLA, RL) — **strong (RL/transformer work lives here)**
6. Data (teleop collection, curation) — **strong (it's pipelines); the real bottleneck**
7. Evaluation / deployment (success rates, failure analysis, video) — **strong**
8. Task orchestration (language → plan → skills) — **strong (LLM background)**

The half I'm good at — 3, 5, 6, 7, 8 — is exactly the half foundation-model robotics
companies hire for. The plan spends most of its time there while forcing just enough
hardware contact (M0–M3) to close the 1–2 / 4 gap.

## Practical notes

- **Compute:** training on the MacBook (MPS) is slow. Rent a cloud GPU for ACT / Diffusion /
  VLA training runs; keep the Mac for teleop, data collection, and eval. Env is the
  `lerobot-experiments` conda venv (Py 3.12).
- **Moving target:** at M4, check the latest SmolVLA / π release before committing — the
  open-VLA frontier shifts monthly.
