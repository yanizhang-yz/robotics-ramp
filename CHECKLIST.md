# 90-Day Robotics Ramp — Weekly Checklist

Track your progress. Check items off as you complete them. Commit this file to your repo and update it weekly — it doubles as a public progress log that recruiters will see.

**Start date:** _____________
**Target end date:** _____________
**Hours/week target:** _____________

---

## Month 1: Foundations + First Demo

### Week 1 — Environment + PyTorch refresher
- [ ] Install Python 3.11, set up uv or conda environment
- [ ] Install PyTorch (with MPS for Apple Silicon)
- [ ] Create public GitHub repo `robotics-ramp` with intro README
- [ ] Complete PyTorch fundamentals (MNIST feedforward classifier)
- [ ] Build CNN on CIFAR-10
- [ ] Build small RNN on character-level Shakespeare
- [ ] Implement gradient descent from scratch
- [ ] Implement backprop from scratch in NumPy (consolidation day)
- [ ] Polish repo: README highlights, notebook intros, screenshots
- [ ] Write Week 1 reflection (`week-01/REFLECTION.md`)
- **Robotics Watch:** π0.5 demo video (Physical Intelligence blog) + Sergey Levine talk on YouTube
- **Deliverable:** Three working notebooks + two from-scratch scripts + reflection

### Week 2 — Build a transformer from scratch
- [ ] Watch Karpathy "Let's build GPT: from scratch" (~2 hrs, code along)
- [ ] Watch Karpathy "Let's build the GPT Tokenizer" (~2 hrs, code along)
- [ ] Implement your own ~300-line transformer in PyTorch
- [ ] Train it on a dataset of your choice
- [ ] Generate samples, document them in README
- [ ] Write blog post: "What I learned implementing a transformer"
- **Robotics Watch:** Skim π0 paper / blog post (no math, just architecture diagrams). Note where you see transformers used.
- **Deliverable:** Working transformer in repo with training samples + 1 blog post

### Week 3 — RL fundamentals + first robotics paper
- [ ] CS224R Lecture 1 (Class intro, MDPs, why deep RL)
- [ ] CS224R Lecture 2 (Imitation learning intro)
- [ ] CS224R Lecture 3 (Imitation learning advanced)
- [ ] CS224R Lecture 4 (Policy gradients intro)
- [ ] Create `paper-notes/` folder in repo
- [ ] Read ALOHA paper (Zhao et al.)
- [ ] Write 1-paragraph summary of ALOHA in `paper-notes/aloha.md`
- **Robotics Watch:** Boston Dynamics Atlas demo + Agility Robotics Digit warehouse demo. Compare in 1 paragraph.
- **Deliverable:** Lecture notes + first paper summary committed

### Week 4 — First simulated robot
- [ ] Install LeRobot (`pip install lerobot`)
- [ ] Complete LeRobot getting-started tutorial
- [ ] Run a pretrained policy in MuJoCo or sim
- [ ] Fine-tune diffusion policy on pusht example from scratch
- [ ] Capture video of trained policy working
- [ ] Write `month-1-recap.md` blog post draft
- **Robotics Watch:** Read ALOHA paper's introduction and methods sections (skip math). Notice the architecture overlap with what you've built.
- **Deliverable:** Trained policy + video + write-up in repo

**End of Month 1 self-check:**
- [ ] Can read a robotics paper and follow most of it
- [ ] Can train a model end-to-end
- [ ] Have a simulated robot doing something from your code

---

## Month 2: Real Robot + Real Literacy

### Week 5 — Order hardware, deepen RL, more papers
- [ ] Order SO-101 arm (or ALOHA setup if budget allows)
- [ ] CS224R Lecture 5 (Policy gradients advanced)
- [ ] CS224R Lecture 6 (Actor-critic methods)
- [ ] CS224R Lecture 7 (Value-based methods / Q-learning)
- [ ] CS224R Lecture 8 (Model-based RL)
- [ ] CS224R Lecture 9 (Offline RL intro)
- [ ] CS224R Lecture 10 (Offline RL advanced)
- [ ] Read Diffusion Policy paper, write summary
- [ ] Read ACT paper, write summary
- [ ] Read RT-2 paper, write summary
- **Robotics Watch:** 3Blue1Brown "Essence of Linear Algebra" series (~3 hrs over the week) — refresh matrix intuition before capstone work.
- **Deliverable:** 3 more paper notes + lecture progress documented

### Week 6 — Hardware setup + teleoperation
- [ ] Hardware arrived
- [ ] Assemble robot arm (document with photos)
- [ ] Install drivers + firmware
- [ ] Calibrate the arm
- [ ] Get teleoperation working (leader-follower or keyboard)
- [ ] Record video of yourself teleoperating
- [ ] Write up setup process including failures in repo
- **Robotics Watch:** V-JEPA 2 paper + Cosmos paper (Nvidia world model) — skim, focus on the "video pre-training → action fine-tuning" pipeline (the WAM idea).
- **Deliverable:** Teleoperation video + setup guide committed

### Week 7 — Data collection + first real fine-tune + classical robotics literacy
- [ ] Pick a specific visual task (write it down precisely)
- [ ] Set up your workspace + objects
- [ ] Collect 50 teleoperation demonstrations
- [ ] Inspect data quality, identify bad demos
- [ ] Fine-tune ACT or diffusion policy on your data via LeRobot
- [ ] Evaluate policy — record success rate over 20 trials
- [ ] Iterate at least once: more data OR different policy
- [ ] Capture final video + write up results honestly
- [ ] **Classical robotics literacy module (2 hrs):** Forward kinematics intuition, what a Jacobian is, what SLAM and motion planning are at a high level. No coding — just literacy.
- **Robotics Watch:** Kevin Wood's robotics roadmap video — for awareness of the traditional stack you're choosing not to specialize in.
- **Deliverable:** Real robot doing your task + video + honest write-up

### Week 8 — Read π0, π0.5, OpenVLA + start blogging
- [ ] Read π0 paper / blog post, write summary
- [ ] Read π0.5 paper / blog post, write summary
- [ ] Read OpenVLA paper, write summary
- [ ] Read Mobile ALOHA paper, write summary
- [ ] Set up Substack OR `blog/` folder in GitHub
- [ ] Write blog post: "What I learned training a robot policy from scratch"
  - [ ] Aim for ~1500 words
  - [ ] Be honest about what was hard
  - [ ] Include videos/screenshots
- [ ] Publish blog post, share on LinkedIn + X/Twitter
- **Robotics Watch:** Find a recent (2025-2026) talk by Karol Hausman or Chelsea Finn from Pi — both teach CS224R, both lead Pi. Pick one 30-min talk.
- **Deliverable:** 4 paper notes + 1 substantive blog post live

**End of Month 2 self-check:**
- [ ] Have a real robot doing a real task because of your work
- [ ] Can hold a 30-min conversation about VLAs without bluffing
- [ ] Public footprint exists (GitHub + blog post)

---

## Month 3: Specialize, Build Capstone, Get the Job

### Week 9 — Pick specialization + scope capstone
- [ ] Review last 8 weeks: what pulled you in most?
- [ ] Pick ONE path:
  - [ ] Path A — Infrastructure / data pipeline
  - [ ] Path B — Sim-to-real / evaluation harness
  - [ ] Path C — Inference / deployment optimization
- [ ] Write capstone design doc (1-2 pages):
  - [ ] Problem statement
  - [ ] Why it matters to deployment-focused robotics companies
  - [ ] Architecture sketch
  - [ ] Success metrics
  - [ ] Stretch goals
- [ ] Set up new repo or sub-directory for capstone
- [ ] First commits: skeleton structure + dependencies
- **Robotics Watch:** Find one recent talk from your target capstone's domain (Pi infra talk for Path A; reliability/eval talk from Figure or Skild for Path B; Nvidia GR00T inference talk for Path C).
- **Deliverable:** Capstone scoped, design doc committed, project skeleton in place

### Week 10 — Build capstone (week 1 of 2)
- [ ] Daily commits showing progress
- [ ] Core functionality working end-to-end on small scale
- [ ] Hit your first major milestone from the design doc
- [ ] Document tradeoffs and decisions as you go
- [ ] Update design doc with reality checks
- **Robotics Watch:** Skip this week — focus is execution. Capstone is your robotics watch.
- **Deliverable:** Capstone half-built, working end-to-end on small scale

### Week 11 — Finish capstone + start outreach
- [ ] Polish capstone:
  - [ ] Clean README with architecture diagram
  - [ ] Video demo (Loom or YouTube)
  - [ ] Honest "limitations" section
  - [ ] Reproduction instructions someone else could follow
- [ ] Write blog post #2 about capstone (~2000 words)
- [ ] Make outreach list: 15-20 people in target companies (LinkedIn search)
- [ ] Send 5 outreach messages this week (template in `outreach.md`)
- [ ] Update your LinkedIn headline + about section
- **Robotics Watch:** Read recent posts from Pi, Skild, Figure, 1X — get a sense of the public voice of each company so your outreach can reference it specifically.
- **Deliverable:** Capstone shipped, blog #2 live, 15+ contacts, 3+ replies

### Week 12 — Apply, interview, iterate
- [ ] Update resume (robotics work at top, SWE history below)
- [ ] Tier 1 applications:
  - [ ] Figure AI
  - [ ] Skild AI
  - [ ] 1X Technologies
  - [ ] Agility Robotics
- [ ] Tier 2 applications:
  - [ ] Physical Intelligence
  - [ ] Nvidia Robotics / GR00T
  - [ ] Boston Dynamics AI Institute
  - [ ] Amazon Robotics (ex-Covariant team)
- [ ] Continue outreach: 5+ more messages this week
- [ ] Interview prep:
  - [ ] Systems design refresh (your strength)
  - [ ] ML systems design practice problems
  - [ ] Practice 5-min capstone walkthrough out loud
  - [ ] Practice "why robotics now" story
- [ ] Track applications + outreach in spreadsheet
- **Robotics Watch:** Watch one recent VLA-vs-WAM debate clip (e.g., LeCun talk, 1X Bernt talk) so you have a current view for interview discussion.
- **Deliverable:** 8+ applications submitted, 2+ warm intros, first phone screens scheduled

---

## Anti-failure-mode reminders

- **Ship ugly code in week 1.** Don't wait until you "know enough."
- **Public commits are the forcing function.** Push daily even when it's small.
- **Don't binge papers you don't have context for.** If you bounce off, come back later.
- **Hardware will fight you.** Budget extra time in weeks 6-7. Document the fights.
- **Your 8 years of SWE is the asset, not the liability.** Lead with it in interviews.
- **Coffee chats > cold applications.** Warm intros are 10x more effective.
- **Robotics Watch is not optional.** 20-30 min/week of curated viewing keeps the destination visible. Skip it and motivation erodes by Week 5.

## Course choice notes

**Primary RL course: CS224R (Stanford, Chelsea Finn + Karol Hausman).** Spring 2025 lectures on YouTube. Robotics-flavored throughout, taught by Pi co-founders. Heavier emphasis on imitation learning and offline RL — the techniques behind modern VLAs.

**Reference / optional: CS285 (Berkeley, Sergey Levine).** Wider RL coverage, more math-heavy. Useful as a deeper reference if any CS224R topic feels under-explained. Also taught by a Pi co-founder.

Both courses are taught by Pi people. CS224R is the more direct fit; CS285 is the deeper foundation if you want it.

## Budget tracking

- [ ] Hardware (~$300-450): _____________
- [ ] Compute (~$50-100/mo): _____________
- [ ] Other: _____________
- **Total spent:** _____________

## Reflection log

After each week, write 2-3 sentences here on what worked, what didn't, what you'd do differently:

**Week 1:**

**Week 2:**

**Week 3:**

**Week 4:**

**Week 5:**

**Week 6:**

**Week 7:**

**Week 8:**

**Week 9:**

**Week 10:**

**Week 11:**

**Week 12:**
