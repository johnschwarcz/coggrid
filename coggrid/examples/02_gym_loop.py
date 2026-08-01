"""Drive the environment through the Gymnasium API.

Shows a random agent, then an agent that tracks the exact Bayesian posterior
using the likelihood table exposed in ``info``.

    python examples/02_gym_loop.py
"""

import numpy as np

from coggrid import CogGridEnv, CogGridConfig

cfg = CogGridConfig(n_states=300, n_contexts=2, n_realizations=8, n_steps=30, seed=0)


def run(policy, episodes=200, expose_likelihood=False):
    env = CogGridEnv(cfg, expose_likelihood=expose_likelihood, seed=0)
    returns = []
    for _ in range(episodes):
        obs, info = env.reset()
        state = policy.reset(obs, info)
        total = 0.0
        done = False
        while not done:
            action = policy.act(obs, state)
            obs, reward, terminated, truncated, info = env.step(action)
            state = policy.observe(obs, state)
            total += reward
            done = terminated or truncated
        returns.append(total / cfg.n_steps)
    return float(np.mean(returns))


class Random:
    def reset(self, obs, info):
        self.rng = getattr(self, "rng", np.random.default_rng(0))
        return None

    def act(self, obs, state):
        return int(self.rng.integers(cfg.n_realizations))

    def observe(self, obs, state):
        return state


class BayesOptimal:
    """Exact posterior over the joint realisation, updated online."""

    def reset(self, obs, info):
        self.rates = info["rates"]            # (n_obs, R, ..., R)
        self.goal = obs["goal_context"]
        log_p, log_q = np.log(self.rates), np.log1p(-self.rates)
        self.log_p, self.log_q = log_p, log_q
        self.log_post = np.zeros(self.rates.shape[1:])
        return None

    def observe(self, obs, state):
        bits = obs["observation"].astype(float)
        self.log_post += np.einsum("o,o...->...", bits, self.log_p) + np.einsum(
            "o,o...->...", 1 - bits, self.log_q
        )
        return state

    def act(self, obs, state):
        axes = tuple(a for a in range(self.log_post.ndim) if a != self.goal)
        marginal = np.exp(self.log_post - self.log_post.max())
        if axes:
            marginal = marginal.sum(axis=axes)
        return int(marginal.argmax())


print(f"random      mean per-step reward: {run(Random()):.3f}")
print(f"bayes-exact mean per-step reward: {run(BayesOptimal(), expose_likelihood=True):.3f}")
print(f"chance level:                     {1 / cfg.n_realizations:.3f}")
