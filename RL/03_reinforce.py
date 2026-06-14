"""
Week 3 · RL foundations — REINFORCE (vanilla policy gradient) on CartPole, in PyTorch.

Goal: feel the log-prob policy gradient — the first NN-based RL. A policy network
outputs action probabilities; we sample, run a full episode, and push up the log-prob
of the actions taken, weighted by their (normalized) return.

Build-from-blank: fill the TODOs, run, and watch the running reward climb. CartPole-v1
is "solved" around an average of 475 (max episode length is 500). When stuck, diff
against CleanRL's REINFORCE.

Deps: pip install "gymnasium[classic-control]" torch numpy
"""

import gymnasium as gym
import torch
import torch.nn as nn

GAMMA = 0.99
LR = 1e-2
EPISODES = 1000


class Policy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        # TODO 1 — define the network:
        #   Linear(obs_dim, hidden) -> ReLU -> Linear(hidden, n_actions)
        # The output is raw logits, one per action.
        ...

    def forward(self, x):
        # TODO 1 (cont.) — return the action logits for input x.
        ...


def compute_returns(rewards, gamma=GAMMA):
    # TODO 2 — discounted return at each timestep, computed backwards:
    #   G_t = r_t + gamma * G_{t+1}
    # Return them as a float tensor, then normalize: (G - mean) / (std + 1e-8).
    ...


def train():
    env = gym.make("CartPole-v1")
    policy = Policy(env.observation_space.shape[0], env.action_space.n)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    running = 0.0

    for ep in range(EPISODES):
        s, _ = env.reset()
        log_probs, rewards = [], []
        done = False
        while not done:
            logits = policy(torch.as_tensor(s, dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            log_probs.append(dist.log_prob(a))
            s, r, term, trunc, _ = env.step(a.item())
            rewards.append(r)
            done = term or trunc

        returns = compute_returns(rewards)
        log_probs = torch.stack(log_probs)

        # TODO 3 — policy-gradient loss. We want to MAXIMIZE sum(log_prob * return),
        # so we minimize its negative:
        #   loss = -(log_probs * returns).sum()
        loss = ...

        opt.zero_grad()
        loss.backward()
        opt.step()

        running = 0.05 * sum(rewards) + 0.95 * running
        if ep % 50 == 0:
            print(f"ep {ep:4d}  running reward {running:6.1f}")

    return running


if __name__ == "__main__":
    final = train()
    print(f"final running reward: {final:.1f}")
    # REINFORCE is high-variance, so this is a "did it learn at all" sanity bar, not a
    # hard pass/fail. A correct version comfortably clears 195 and usually nears 500.
    assert final > 150, f"running reward {final:.1f} is low — check the return/loss signs"
    print("\u2713 REINFORCE is learning CartPole")
