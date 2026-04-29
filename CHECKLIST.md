# 90-Day Robotics Ramp — Weekly Checklist

Track your progress. Check items off as you complete them. Commit this file to your repo and update it weekly — it doubles as a public progress log that recruiters will see.

**Start date:** _____________
**Target end date:** _____________
**Hours/week target:** _____________

---

## Month 1: Foundations + First Demo

### Week 1 — Environment + PyTorch refresher
- [ ] Install Python 3.11, set up uv or conda environment
- [ ] Install PyTorch (with CUDA if you have an NVIDIA GPU)
- [ ] Create public GitHub repo `robotics-ramp` with intro README
- [ ] Complete PyTorch 60-minute blitz tutorial
- [ ] Build feedforward net on MNIST (commit notebook)
- [ ] Build CNN on CIFAR-10 (commit notebook)
- [ ] Build small RNN on character-level text (commit notebook)
- [ ] Update README with what you learned this week
- **Deliverable:** Three working notebooks committed with clean READMEs

### Week 2 — Build a transformer from scratch
- [ ] Watch Karpathy "Let's build GPT: from scratch" (~2 hrs, code along)
- [ ] Watch Karpathy "Let's build the GPT Tokenizer" (~2 hrs, code along)
- [ ] Implement your own ~300-line transformer in PyTorch
- [ ] Train it on a dataset of your choice
- [ ] Generate samples, document them in README
- **Deliverable:** Working transformer in repo with training samples

### Week 3 — RL fundamentals + first robotics paper
- [ ] CS285 Lecture 1 (intro)
- [ ] CS285 Lecture 2 (supervised learning of behaviors / imitation learning)
- [ ] CS285 Lecture 3 (PyTorch tutorial / intro to RL)
- [ ] CS285 Lecture 4 (intro to RL)
- [ ] Create `paper-notes/` folder in repo
- [ ] Read ALOHA paper (Zhao et al.)
- [ ] Write 1-paragraph summary of ALOHA in `paper-notes/aloha.md`
- **Deliverable:** Lecture notes + first paper summary committed

### Week 4 — First simulated robot
- [ ] Install LeRobot (`pip install lerobot`)
- [ ] Complete LeRobot getting-started tutorial
- [ ] Run a pretrained policy in MuJoCo or sim
- [ ] Fine-tune diffusion policy on pusht example from scratch
- [ ] Capture video of trained policy working
- [ ] Write `month-1-recap.md` blog post draft (lessons + screenshots)
- **Deliverable:** Trained policy + video + write-up in repo

**End of Month 1 self-check:**
- [ ] Can read a robotics paper and follow most of it
- [ ] Can train a model end-to-end
- [ ] Have a simulated robot doing something from your code

---

## Month 2: Real Robot + Real Literacy

### Week 5 — Order hardware, deepen RL, more papers
- [ ] Order SO-101 arm (or ALOHA setup if budget allows)
- [ ] CS285 Lecture 5 (policy gradients)
- [ ] CS285 Lecture 6 (actor-critic)
- [ ] CS285 Lecture 7 (value function methods)
- [ ] CS285 Lecture 8 (deep RL with Q-functions)
- [ ] CS285 Lecture 9 (advanced policy gradients)
- [ ] CS285 Lecture 10 (optimal control & planning)
- [ ] Read Diffusion Policy paper, write summary
- [ ] Read ACT paper, write summary
- [ ] Read RT-2 paper, write summary
- **Deliverable:** 3 more paper notes + lecture progress documented

### Week 6 — Hardware setup + teleoperation
- [ ] Hardware arrived
- [ ] Assemble robot arm (document with photos)
- [ ] Install drivers + firmware
- [ ] Calibrate the arm
- [ ] Get teleoperation working (leader-follower or keyboard)
- [ ] Record video of yourself teleoperating
- [ ] Write up setup process including failures in repo
- **Deliverable:** Teleoperation video + setup guide committed

### Week 7 — Data collection + first real fine-tune
- [ ] Pick a specific visual task (write it down precisely)
- [ ] Set up your workspace + objects
- [ ] Collect 50 teleoperation demonstrations
- [ ] Inspect data quality, identify bad demos
- [ ] Fine-tune ACT or diffusion policy on your data via LeRobot
- [ ] Evaluate policy — record success rate over 20 trials
- [ ] Iterate at least once: more data OR different policy
- [ ] Capture final video + write up results honestly
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
- **Deliverable:** Capstone scoped, design doc committed, project skeleton in place

### Week 10 — Build capstone (week 1 of 2)
- [ ] Daily commits showing progress
- [ ] Core functionality working end-to-end on small scale
- [ ] Hit your first major milestone from the design doc
- [ ] Document tradeoffs and decisions as you go
- [ ] Update design doc with reality checks
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
- **Deliverable:** 8+ applications submitted, 2+ warm intros, first phone screens scheduled

---

## Anti-failure-mode reminders

- **Ship ugly code in week 1.** Don't wait until you "know enough."
- **Public commits are the forcing function.** Push daily even when it's small.
- **Don't binge papers you don't have context for.** If you bounce off, come back later.
- **Hardware will fight you.** Budget extra time in weeks 6-7. Document the fights.
- **Your 8 years of SWE is the asset, not the liability.** Lead with it in interviews.
- **Coffee chats > cold applications.** Warm intros are 10x more effective.

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
