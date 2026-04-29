# Capstone Project Designs — Three Specialization Paths

You'll pick one of these in Week 9. Each is scoped to be buildable in ~3 weeks of focused work, impressive enough to show in interviews, and aligned with what deployment-focused robotics companies actually need. Read all three before deciding.

---

## Path A — Infrastructure / Data Pipeline

**Project name suggestion:** `teleop-pipeline` or `policy-factory`

### What you build

A small but real distributed system that ingests teleoperation demonstrations, processes them into training data, runs a nightly fine-tuning job on a fresh policy, evaluates it automatically, and surfaces metrics in a dashboard. End-to-end, the kind of system every robotics company needs but few have polished.

### Why this matters for hiring

Every VLA company is data-bottlenecked, not model-bottlenecked. Pi, Figure, Skild, 1X — all of them have armies of teleoperators producing terabytes of demonstration data per week, and the infrastructure to turn that data into trained policies is genuinely the bottleneck. Most ML researchers can't build this. Most SWEs don't know enough robotics to build it well. You can be both. This is the highest-leverage path for your background.

### Architecture

```
[ Teleop sessions ] -> [ Object store (S3/local) ]
                              |
                              v
                     [ Ingestion service ]
                     - Validates schema
                     - Computes data quality metrics
                     - Writes to versioned dataset
                              |
                              v
                     [ Training orchestrator ]
                     - Cron-triggered nightly
                     - Spins up GPU job (Modal/RunPod)
                     - Logs to W&B
                              |
                              v
                     [ Evaluation harness ]
                     - Runs trained policy in sim across N variations
                     - Records success rate
                              |
                              v
                     [ Dashboard ]
                     - Streamlit/FastAPI + simple frontend
                     - Shows: data growth, training runs, eval over time
```

### Concrete tech stack

- **Storage:** Local filesystem with parquet files for v1, optionally MinIO or S3 for v2. Use LeRobot's dataset format (it's standard now in the open-source community).
- **Orchestration:** Prefect or Dagster for the pipeline (don't overengineer with Airflow). A simple cron job is also fine for v1.
- **Training:** PyTorch + LeRobot's training scripts as the base, run on Modal or RunPod ($1-3/hr for an A100). Your code wraps theirs.
- **Tracking:** Weights & Biases free tier (industry standard, recruiters recognize it).
- **Evaluation:** MuJoCo + a small set of variation scripts you write (different lighting, object positions, distractor objects).
- **Dashboard:** Streamlit is fastest to ship. FastAPI + a Next.js frontend if you want to flex web-dev chops.

### 3-week build plan

**Week 10, days 1-3:** Ingestion + storage. Build the data validator. Take your own 50 demos from week 7 and a few open-source datasets from Hugging Face Hub, get them all into your unified format. Compute basic quality metrics (episode length, success/failure flag, action variance).

**Week 10, days 4-7:** Training orchestrator. Write the script that pulls latest data, kicks off a fine-tuning job on Modal, logs results. Run it manually a few times. Then put it on a cron schedule.

**Week 11, days 1-4:** Evaluation harness. Write the variation generator (10-20 different sim setups for your task). Run policies through it, record results, write to a database (SQLite is fine).

**Week 11, days 5-7:** Dashboard + polish. Streamlit dashboard showing pipeline status, data growth over time, training run history, eval scores. Architecture diagram, README, demo video.

### Stretch goals

- Add a "data quality scorer" that flags low-quality demos automatically before training
- Add policy comparison (run two policies on identical eval set, show win rate)
- Add a "what changed" diff between training runs

### Honest tradeoffs to discuss in interviews

- You're not training at scale, so distributed training tradeoffs are theoretical for you — don't overclaim.
- Your evaluation set is small, so noise dominates. Discuss this openly.
- The system is single-node simulating a multi-team workflow — be clear about what scaling would actually require.

### What this signals to a hiring manager

"This person can build the boring infrastructure that makes the cool research possible. They understand data quality matters more than models. They've thought about evaluation. They've shipped systems before. We need them on the data team / training infra team."

### Best target companies for this capstone

- Physical Intelligence (data + training infra roles)
- Skild AI (commercialization-focused, needs pipelines)
- Figure AI (data engineering team)
- 1X Technologies (heavy teleop data collection)
- Amazon Robotics

---

## Path B — Sim-to-Real / Evaluation Harness

**Project name suggestion:** `policy-stress-test` or `robust-eval`

### What you build

A systematic evaluation harness that takes any trained policy and stress-tests it across hundreds of controlled variations of a task in simulation, identifies failure modes, and produces a robustness report. Then you do something most people don't: you also test the same policy on the real robot to measure the sim-to-real gap, and you analyze which simulation variations actually predict real-world failure.

### Why this matters for hiring

This is the dirty secret of robotics ML: most policies are evaluated on a tiny test set in conditions that match training, success is reported as 80%+, and then they fail in the real world. Companies *know* this is a problem and pay engineers specifically to solve it. Skild AI and Figure both have teams dedicated to evaluation. The sim-to-real gap is one of the field's biggest open problems, so building tooling around it puts you on the right side of where the puck is going.

### Architecture

```
[ Trained policy ] + [ Task config ]
            |
            v
[ Variation generator ]
- Lighting (intensity, angle, color temp)
- Object positions (sampled from distribution)
- Object appearance (textures, colors)
- Distractors (random extra objects)
- Camera pose perturbations
- Initial robot state perturbations
            |
            v
[ Sim eval runner ]
- Parallel rollouts in MuJoCo
- Records: success/fail, time-to-completion, trajectory
            |
            v
[ Real eval runner ]
- Same task on physical robot
- 20-50 trials with controlled variations
            |
            v
[ Analysis + report ]
- Failure mode clustering
- Sim-real gap quantification
- Robustness score
- Suggested data collection priorities
```

### Concrete tech stack

- **Simulation:** MuJoCo (LeRobot uses gym-style envs that wrap MuJoCo). Isaac Sim if you want to flex on a more industry-relevant tool, but MuJoCo is faster to iterate.
- **Variation:** Custom Python with `domain_randomization` patterns. Hydra for config management.
- **Parallel rollouts:** Python multiprocessing for v1. Ray if you want to flex.
- **Analysis:** pandas + scikit-learn for clustering failure modes. Plotly for interactive failure visualizations.
- **Report:** Auto-generated HTML report (Jinja2 templating) + a notebook for deep-dive analysis.

### 3-week build plan

**Week 10, days 1-3:** Variation generator. Pick 5-6 axes of variation for your week-7 task. Write generators that sample configurations. Build a sim env that consumes them.

**Week 10, days 4-7:** Sim eval runner. Run 200+ rollouts of your week-7 policy. Record everything. First analysis: which variations break it most?

**Week 11, days 1-3:** Real eval runner. This is the painful part — run your robot 30-50 times with controlled variations matching what you tested in sim. Record success/failure honestly.

**Week 11, days 4-5:** Sim-real gap analysis. For each variation axis, do sim and real predictions correlate? Where do they diverge? This analysis is the gold.

**Week 11, days 6-7:** Report generator + polish. HTML report template, README, demo video showing the worst failure modes.

### Stretch goals

- Train a *failure predictor* — a small model that takes (sim variation params) and predicts whether real-world will fail
- Use the failure modes to suggest a targeted data collection plan ("collect 20 more demos with these conditions")
- Compare two different policies on the same harness — which is more robust?

### Honest tradeoffs

- Your sample size on real robot evaluation is small (~30-50). Statistical significance is shaky. Acknowledge this.
- Sim-to-real gap analysis from one task on one robot doesn't generalize. Be careful with claims.
- You're not solving the gap — you're measuring it. That's actually fine, measuring is the harder part.

### What this signals to a hiring manager

"This person actually understands that robot policies fail in messy ways. They can build evaluation tooling. They have hardware experience. They think rigorously about what 'success' means. We need them on the eval team or on a deployment team that cares about reliability."

### Best target companies for this capstone

- Figure AI (deployment in real factories)
- Agility Robotics (warehouse deployments, reliability-critical)
- Skild AI (commercializing, reliability-focused)
- Amazon Robotics
- Boston Dynamics AI Institute (research-meets-deployment)

---

## Path C — Inference / Deployment Optimization

**Project name suggestion:** `vla-on-edge` or `fast-pi`

### What you build

You take an open-source VLA (OpenVLA or π0 weights), get it running, and then systematically optimize it for fast inference on edge hardware (Jetson Orin if you can get one, or just your laptop GPU as v1). You quantize it, profile it, find bottlenecks, and benchmark the speed-vs-quality tradeoffs honestly. The output is a polished benchmark report and a deployment-ready inference server.

### Why this matters for hiring

VLAs are big — π0 is around 3B parameters, others are larger. Robot control needs sub-100ms latency to feel responsive, ideally 30Hz+ for smooth motion. Bridging that gap is *brutally* hard and is where the rubber meets the road for actually shipping these models. Companies need engineers who understand both the model side and the systems side.

### Architecture

```
[ Open-source VLA weights (OpenVLA / π0) ]
            |
            v
[ Baseline inference ] -> measure: latency, memory, accuracy on test set
            |
            v
[ Optimization passes ]
- FP16 -> INT8 quantization (bitsandbytes / GPTQ)
- KV-cache for autoregressive components
- Batching / async inference
- ONNX or TensorRT export
            |
            v
[ Benchmarking suite ]
- Latency at different batch sizes
- Memory at different precisions
- Accuracy degradation per optimization
            |
            v
[ Deployment server ]
- FastAPI service that wraps optimized model
- Streaming action chunks
- Health checks, metrics
            |
            v
[ Hardware tests ]
- Laptop GPU baseline
- Jetson Orin (if accessible)
- CPU-only fallback
```

### Concrete tech stack

- **Base model:** OpenVLA is best documented for this. π0 weights are open via openpi but more sensitive to setup. Pick one.
- **Quantization:** bitsandbytes for quick wins, GPTQ for more sophisticated. Hugging Face's `optimum` library helps.
- **Serving:** FastAPI for the wrapper. Nvidia Triton if you want to flex on industry-standard serving.
- **Profiling:** PyTorch profiler, nvidia-smi, py-spy. Document everything you find.
- **Edge hardware:** Jetson Orin Nano (~$500) is the obvious target. If budget is tight, just compare laptop GPU vs CPU.

### 3-week build plan

**Week 10, days 1-3:** Get baseline working. Clone OpenVLA repo, get inference running on a single example, measure latency and memory. This week is mostly fighting environment setup — don't underestimate.

**Week 10, days 4-7:** First optimizations. FP16 inference, then INT8 quantization. Re-measure latency, memory, and accuracy on a small test set. Document the regressions honestly.

**Week 11, days 1-3:** Serving infrastructure. FastAPI wrapper with proper async handling, action chunk streaming, basic monitoring.

**Week 11, days 4-5:** Hardware comparison. Run benchmarks on every device you have access to. If you have a Jetson, that's the headline result. If not, laptop GPU vs CPU is fine.

**Week 11, days 6-7:** Benchmark report + polish. Tables of latency-quality tradeoffs, charts, README with reproduction steps.

### Stretch goals

- Add speculative decoding for the autoregressive parts
- Try TensorRT export for the largest speedup (and the largest pain)
- Build a tiny demo: live camera feed -> VLA -> action prediction at 30Hz

### Honest tradeoffs

- Quantization always degrades quality somewhat. Don't fudge the numbers.
- Without a real robot in the loop, you're optimizing for proxy metrics. Acknowledge this.
- TensorRT and ONNX have nasty edge cases with custom layers. Time-box this — don't lose a week to a single op.

### What this signals to a hiring manager

"This person understands that the model is a means to an end. They can profile, optimize, and serve. They've actually thought about edge deployment. They have systems chops. We need them on the inference / deployment / on-robot software team."

### Best target companies for this capstone

- Figure AI (on-robot inference is critical)
- 1X Technologies (humanoid, real-time control)
- Nvidia Robotics (literally building the inference stack)
- Apptronik
- Any humanoid company — they all need this

---

## How to choose

If you're genuinely torn, here's the decision tree I'd use:

**If your last 8 years involved a lot of distributed systems, data infrastructure, or backend at scale →** Path A. Highest leverage, plays directly to your existing strengths, broadest fit across companies.

**If you found yourself most curious during the hardware weeks (6-7), or you like the puzzle of "why did this break" →** Path B. Suits people who think rigorously and like detective work.

**If you've done performance optimization before, or you find yourself looking at model architectures and thinking "I bet that's slow" →** Path C. Most niche but most defensible — fewer people can do this well.

**If you're still torn:** Path A. It has the broadest applicability, it's the most obviously useful to deployment-focused companies, and you can pivot to B or C in your second role once you've got robotics experience on your resume.

## What every capstone needs regardless of path

- A 1-2 minute demo video at the top of the README. Recruiters watch videos; they often don't read.
- An architecture diagram. Use Excalidraw or draw.io. Even a hand-drawn one is fine.
- An honest "limitations" section. This is the single thing that separates portfolio projects from senior-engineer projects.
- A 2000-word blog post explaining the *why* behind your design decisions. Cross-post to your Substack/Medium and share on LinkedIn.
- Reproduction instructions someone else could actually follow. Test them by having a friend try.

## What no capstone needs

- A cute name with bad puns
- A landing page
- More than 3 weeks of work — past that you're polishing instead of shipping
- "Cutting-edge" tech for its own sake (no, you don't need to use Mojo or Triton kernels)

The capstone is the single most important artifact in your job search. Spend the time it deserves and no more.
