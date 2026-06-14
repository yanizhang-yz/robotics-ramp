"""
Week 3 · RL foundations — Tabular Q-learning on FrozenLake (model-free).

Goal: feel the model-free TD update. No transition model now — the agent learns
Q[s, a] purely by interacting, with ε-greedy exploration.

Build-from-blank: fill the TODOs, run the file. When stuck, diff against CleanRL or
any tabular Q-learning reference.

Deps: pip install "gymnasium[toy-text]" numpy
"""

import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)
nS, nA = env.observation_space.n, env.action_space.n

ALPHA = 0.1        # learning rate
GAMMA = 0.99
EPISODES = 5000
EPS_START, EPS_END = 1.0, 0.05


def epsilon_greedy(Q, s, eps):
    # TODO 1 — with probability eps return a uniformly random action,
    #          otherwise return argmax_a Q[s, a].
    return 0  # <-- replace


def train():
    Q = np.zeros((nS, nA))
    for ep in range(EPISODES):
        eps = max(EPS_END, EPS_START - ep / EPISODES)   # linear decay
        s, _ = env.reset()
        done = False
        while not done:
            a = epsilon_greedy(Q, s, eps)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc

            # TODO 2 — Q-learning TD update:
            #   target = r + GAMMA * max_a' Q[s2, a']   (use just r if `term` is True)
            #   Q[s, a] += ALPHA * (target - Q[s, a])
            pass

            s = s2
    return Q


def eval_policy(Q, episodes=200):
    wins = 0
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        while not done:
            s, r, term, trunc, _ = env.step(int(np.argmax(Q[s])))
            done = term or trunc
            wins += int(r == 1.0)
    return wins / episodes


if __name__ == "__main__":
    Q = train()
    print("greedy policy (0=Left, 1=Down, 2=Right, 3=Up):")
    print(np.argmax(Q, axis=1).reshape(4, 4))

    success = eval_policy(Q)
    print(f"success rate: {success:.2f}")

    assert success > 0.7, f"expected >0.7 success, got {success:.2f} — check the TD update"
    print("\u2713 Q-learning learned a good policy")
