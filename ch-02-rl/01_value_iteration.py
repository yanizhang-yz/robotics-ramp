"""
Week 3 · RL foundations — Value Iteration on FrozenLake (tabular, model-based).

Goal: feel the Bellman optimality backup. No neural nets here — just an MDP with a
known transition model (env.P) and a value table V[s].

Build-from-blank: fill the TODOs, then run the file. The asserts at the bottom tell
you when it's right. When stuck, compare against any standard value-iteration ref.

Deps: pip install "gymnasium[toy-text]" numpy
"""

import gymnasium as gym
import numpy as np

# is_slippery=False => deterministic, so the optimal policy is a crisp path and the
# asserts are exact. Flip to True later to watch value iteration handle stochasticity.
env = gym.make("FrozenLake-v1", is_slippery=False)
P = env.unwrapped.P            # P[s][a] = list of (prob, next_state, reward, terminated)
nS = env.observation_space.n
nA = env.action_space.n
GAMMA = 0.99


def value_iteration(P, nS, nA, gamma=GAMMA, theta=1e-9):
    V = np.zeros(nS)
    while True:
        delta = 0.0
        for s in range(nS):
            # TODO 1 — Bellman optimal backup for state s.
            #   For each action a, its value is:
            #       q_a = sum over (p, s2, r, done) in P[s][a] of
            #                 p * (r + gamma * V[s2] * (not done))
            #   best = max over a of q_a
            best = 0.0  # <-- replace with max_a q_a
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < theta:
            break

    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        # TODO 2 — greedy policy: policy[s] = argmax over a of q_a (same backup).
        policy[s] = 0  # <-- replace with argmax_a q_a
    return V, policy


def run_policy(policy, episodes=100):
    wins = 0
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        while not done:
            s, r, term, trunc, _ = env.step(int(policy[s]))
            done = term or trunc
            wins += int(r == 1.0)
    return wins / episodes


if __name__ == "__main__":
    V, policy = value_iteration(P, nS, nA)
    print("V (reshaped to the 4x4 grid):")
    print(V.reshape(4, 4).round(3))
    print("policy (0=Left, 1=Down, 2=Right, 3=Up):")
    print(policy.reshape(4, 4))

    success = run_policy(policy)
    print(f"success rate: {success:.2f}")

    # On deterministic FrozenLake the optimal policy reaches the goal every time.
    assert success == 1.0, f"expected 100% success, got {success:.2f} — check the backup"
    print("\u2713 value iteration solved FrozenLake")
