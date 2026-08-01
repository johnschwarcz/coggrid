"""Ideal-observer baselines and the metrics used to score them.

Two observers, both exact given their own assumptions:

``joint``
    Keeps a posterior over the full joint realisation of all active states,
    using :attr:`EpisodeBatch.rates`. This is the Bayes-optimal observer.

``naive``
    Keeps one independent posterior per active state, using
    :attr:`EpisodeBatch.marginal_rates`. It is exactly the joint observer with
    the interaction terms marginalised away, so the difference between the two
    isolates the cost of assuming a factorised latent space — and nothing else.

Inference runs in log space with a cumulative sum, rather than the original's
step-by-step multiply-and-renormalise. The two are mathematically identical;
the log-space version does not underflow when ``n_steps`` is large or the
realisation grid is big.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .world import EpisodeBatch

__all__ = [
    "BeliefTrace",
    "joint_observer",
    "naive_observer",
    "run_observers",
    "factorization_regret",
    "score_belief",
]

ObserverName = Literal["joint", "naive"]

#: Rates are clipped into ``[RATE_FLOOR, 1 - RATE_FLOOR]`` before taking logs.
#: With the default config the rates sit near 0.5 and this never binds, but a
#: large ``likelihood_temp`` can saturate the sigmoid and produce ``log(0)``.
RATE_FLOOR = 1e-12


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BeliefTrace:
    """What an observer believed, and how good that was, at every timestep.

    Attributes
    ----------
    name:
        ``"joint"`` or ``"naive"``.
    belief:
        ``(n_episodes, n_steps, n_contexts, n_realizations)`` marginal posterior
        over each active state's value.
    goal_belief:
        ``(n_episodes, n_steps, n_realizations)`` the slice of ``belief`` for the
        goal state.
    estimate:
        ``(n_episodes, n_steps, n_contexts)`` posterior mean realisation.
    accuracy:
        ``(n_episodes, n_steps)`` 1.0 where the posterior mode equals the true
        goal value.
    p_correct:
        ``(n_episodes, n_steps)`` posterior probability assigned to the true goal
        value. A softer, lower-variance version of ``accuracy``.
    mse:
        ``(n_episodes, n_steps)`` squared error of the posterior mean.
    log_posterior:
        ``(n_episodes, n_steps, ...)`` unnormalised cumulative log-likelihood,
        over the joint realisation grid for the joint observer and over
        ``(n_contexts, n_realizations)`` for the naive one. Kept because it is
        what you need for likelihood-ratio and model-comparison analyses.
    """

    name: ObserverName
    belief: np.ndarray
    goal_belief: np.ndarray
    estimate: np.ndarray
    accuracy: np.ndarray
    p_correct: np.ndarray
    mse: np.ndarray
    log_posterior: np.ndarray

    def mean_curves(self) -> dict[str, np.ndarray]:
        """Episode-averaged learning curves, each ``(n_steps,)``."""
        return {
            "accuracy": self.accuracy.mean(0),
            "p_correct": self.p_correct.mean(0),
            "mse": self.mse.mean(0),
        }

    def final(self) -> dict[str, float]:
            """Scalar summary at the last timestep."""
            return {
                "accuracy": float(self.accuracy[:, -1].mean()),
                "p_correct": float(self.p_correct[:, -1].mean()),
                "mse": float(self.mse[:, -1].mean()),
            }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        f = self.final()
        return (
            f"BeliefTrace({self.name!r}, n_episodes={self.belief.shape[0]}, "
            f"final acc={f['accuracy']:.3f}, p_correct={f['p_correct']:.3f})"
        )


# --------------------------------------------------------------------------- #
# numerics
# --------------------------------------------------------------------------- #
def _logsumexp(x: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    peak = np.max(x, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return peak + np.log(np.exp(x - peak).sum(axis=axis, keepdims=True))


def _cumulative_log_likelihood(
    observations: np.ndarray, rates: np.ndarray
) -> np.ndarray:
    """Accumulate Bernoulli log-likelihood over time.

    ``observations`` is ``(n_eps, n_steps, n_obs)``; ``rates`` is
    ``(n_eps, n_obs, *hypothesis_shape)``. Returns
    ``(n_eps, n_steps, *hypothesis_shape)``.

    Because observations are i.i.d. given the latent state, the posterior after
    ``t`` steps depends only on the running sum of per-step log-likelihoods.
    """
    rates = np.clip(np.asarray(rates, dtype=float), RATE_FLOOR, 1.0 - RATE_FLOOR)
    obs = np.asarray(observations, dtype=float)

    log_p = np.log(rates)
    log_q = np.log1p(-rates)
    per_step = np.einsum("bto,bo...->bt...", obs, log_p) + np.einsum(
        "bto,bo...->bt...", 1.0 - obs, log_q
    )
    return np.cumsum(per_step, axis=1)


# --------------------------------------------------------------------------- #
# observers
# --------------------------------------------------------------------------- #
def joint_observer(batch: EpisodeBatch) -> BeliefTrace:
    """Exact Bayesian inference over the full joint realisation.

    Maintains a posterior over all ``n_realizations ** n_contexts``
    configurations, then marginalises to per-state beliefs for scoring. This is
    the ceiling: no observer can do better on this environment.
    """
    n_contexts = batch.cfg.n_contexts
    log_posterior = _cumulative_log_likelihood(batch.observations, batch.rates)

    config_axes = tuple(range(2, 2 + n_contexts))
    posterior = np.exp(log_posterior - _logsumexp(log_posterior, axis=config_axes))

    # Marginalise the joint posterior down to one distribution per active state.
    belief = np.empty(
        (batch.n_episodes, batch.n_steps, n_contexts, batch.cfg.n_realizations)
    )
    for c in range(n_contexts):
        other = tuple(2 + i for i in range(n_contexts) if i != c)
        belief[:, :, c, :] = posterior.sum(axis=other) if other else posterior

    return _finish("joint", belief, log_posterior, batch)


def naive_observer(batch: EpisodeBatch) -> BeliefTrace:
    """Bayesian inference under the (false) assumption of independent states.

    Uses the marginalised likelihood and normalises each active state's
    posterior separately. Its shortfall against :func:`joint_observer` is the
    price of a factorised representation.
    """
    log_posterior = _cumulative_log_likelihood(
        batch.observations, batch.marginal_rates
    )
    # Normalise over realisations only, independently for each active state.
    belief = np.exp(log_posterior - _logsumexp(log_posterior, axis=-1))
    return _finish("naive", belief, log_posterior, batch)


def _finish(
    name: ObserverName,
    belief: np.ndarray,
    log_posterior: np.ndarray,
    batch: EpisodeBatch,
) -> BeliefTrace:
    metrics = score_belief(belief, batch)
    return BeliefTrace(name=name, belief=belief, log_posterior=log_posterior, **metrics)


def run_observers(batch: EpisodeBatch) -> dict[ObserverName, BeliefTrace]:
    """Run both baselines. Returns ``{"joint": ..., "naive": ...}``."""
    return {"joint": joint_observer(batch), "naive": naive_observer(batch)}


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_belief(belief: np.ndarray, batch: EpisodeBatch) -> dict[str, np.ndarray]:
    """Score a belief array against the ground truth in ``batch``.

    Works on anything shaped ``(n_episodes, n_steps, n_contexts,
    n_realizations)`` — the built-in observers, but equally a learned agent's
    output, which is the point of keeping it a free function.

    Returns a dict with ``goal_belief``, ``estimate``, ``accuracy``,
    ``p_correct`` and ``mse``.
    """
    belief = np.asarray(belief, dtype=float)
    expected_tail = (batch.cfg.n_contexts, batch.cfg.n_realizations)
    if belief.ndim != 4 or belief.shape[2:] != expected_tail:
        raise ValueError(
            f"belief must have shape (n_episodes, n_steps, {expected_tail[0]}, "
            f"{expected_tail[1]}), got {belief.shape}"
        )

    realizations = np.arange(batch.cfg.n_realizations)
    episodes = np.arange(belief.shape[0])

    estimate = belief @ realizations                       # (eps, steps, ctx)
    goal_belief = belief[episodes, :, batch.goal_ind, :]   # (eps, steps, real)

    truth = batch.goal_value[:, None]                      # (eps, 1)
    accuracy = (goal_belief.argmax(axis=-1) == truth).astype(float)
    p_correct = np.take_along_axis(goal_belief, truth[..., None], axis=-1).squeeze(-1)
    goal_estimate = estimate[episodes, :, batch.goal_ind]
    mse = (truth - goal_estimate) ** 2

    return {
        "goal_belief": goal_belief,
        "estimate": estimate,
        "accuracy": accuracy,
        "p_correct": p_correct,
        "mse": mse,
    }


def factorization_regret(
    joint: BeliefTrace,
    naive: BeliefTrace,
    *,
    symmetric: bool = True,
    eps: float = 1e-15,
) -> np.ndarray:
    """Divergence between the joint and naive posteriors over the goal state.

    Returns ``(n_episodes, n_steps)``. This is how far the factorised observer's
    belief has drifted from the optimal one — large where modelling the
    interaction actually mattered, near zero where the episode happened to be
    separable anyway.

    ``symmetric=True`` (the default, matching the original ``SII`` quantity)
    averages the two KL directions.
    """
    p = np.clip(joint.goal_belief, eps, 1.0 - eps)
    q = np.clip(naive.goal_belief, eps, 1.0 - eps)
    forward = (p * np.log(p / q)).sum(-1)
    if not symmetric:
        return forward
    reverse = (q * np.log(q / p)).sum(-1)
    return (forward + reverse) / 2.0
