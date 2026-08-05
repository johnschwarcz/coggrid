"""Ideal-observer baselines and the metrics used to score them.

Two observers, both exact given their own assumptions:

``joint``
    Keeps a posterior over the full joint realization of all active variables,
    using :attr:`EpisodeBatch.rates`. This is the Bayes-optimal observer.

``naive``
    Keeps one independent posterior per active variable, using
    :attr:`EpisodeBatch.marginal_rates`. It is exactly the joint observer with
    the interaction terms marginalized away, so the difference between the two
    isolates the cost of assuming a factorized latent space — and nothing else.

Inference runs in log space with a cumulative sum rather than a step-by-step
multiply-and-renormalise, so it does not underflow when ``n_steps`` is large or
the realization grid is big.
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
    "disentanglement",
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
        over each active variable's value.
    goal_belief:
        ``(n_episodes, n_steps, n_realizations)`` the slice of ``belief`` for the
        goal variable.
    estimate:
        ``(n_episodes, n_steps, n_contexts)`` posterior mean realization.
    accuracy:
        ``(n_episodes, n_steps)`` 1.0 where the posterior mode equals the true
        goal value.
    p_correct:
        ``(n_episodes, n_steps)`` posterior probability assigned to the true goal
        value. A softer, lower-variance version of ``accuracy``.
    mse:
        ``(n_episodes, n_steps)`` squared error of the posterior mean.
    log_posterior:
        ``(n_episodes, n_steps, ...)`` unnormalized cumulative log-likelihood,
        over the joint realization grid for the joint observer and over
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

    Because observations are i.i.d. given the latent variable, the posterior after
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
    """Exact Bayesian inference over the full joint realization.

    Maintains a posterior over all ``n_realizations ** n_contexts``
    configurations, then marginalizes to per-variable beliefs for scoring. This is
    the ceiling: no observer can do better on this environment.
    """
    n_contexts = batch.cfg.n_contexts
    log_posterior = _cumulative_log_likelihood(batch.observations, batch.rates)

    config_axes = tuple(range(2, 2 + n_contexts))
    posterior = np.exp(log_posterior - _logsumexp(log_posterior, axis=config_axes))

    # Marginalise the joint posterior down to one distribution per active variable.
    belief = np.empty(
        (batch.n_episodes, batch.n_steps, n_contexts, batch.cfg.n_realizations)
    )
    for c in range(n_contexts):
        other = tuple(2 + i for i in range(n_contexts) if i != c)
        belief[:, :, c, :] = posterior.sum(axis=other) if other else posterior

    return _finish("joint", belief, log_posterior, batch)


def naive_observer(batch: EpisodeBatch) -> BeliefTrace:
    """Bayesian inference under the (false) assumption of independent variables.

    Uses the marginalized likelihood and normalizes each active variable's
    posterior separately. Its shortfall against :func:`joint_observer` is the
    price of a factorized representation.
    """
    log_posterior = _cumulative_log_likelihood(
        batch.observations, batch.marginal_rates
    )
    # Normalise over realizations only, independently for each active variable.
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
    symmetric: bool = False,
    eps: float = 1e-15,
) -> np.ndarray:
    """Divergence between the joint and naive posteriors over the goal variable.

    Returns ``(n_episodes, n_steps)``. This is how far the factorized observer's
    belief has drifted from the optimal one — large where modelling the
    interaction actually mattered, near zero where the episode happened to be
    separable anyway.

    The default is the paper's definition (arXiv:2603.27134 §3.1): the forward
    divergence ``D_KL(B_joint || B_naive)`` over the goal variable, weighted by
    the optimal observer's belief. ``symmetric=True`` averages both directions
    instead.
    """
    p = np.clip(joint.goal_belief, eps, 1.0 - eps)
    q = np.clip(naive.goal_belief, eps, 1.0 - eps)
    forward = (p * np.log(p / q)).sum(-1)
    if not symmetric:
        return forward
    reverse = (q * np.log(q / p)).sum(-1)
    return (forward + reverse) / 2.0


# --------------------------------------------------------------------------- #
# dis-entanglement
# --------------------------------------------------------------------------- #
def _log_marginal_belief(trace: BeliefTrace, n_contexts: int) -> np.ndarray:
    """``(n_episodes, n_steps, n_contexts, n_realizations)`` log marginal posterior.

    Read off the stored log posterior rather than ``belief``. Beliefs routinely
    fall below 1e-12 once an observer is confident, and dividing two such numbers
    to recover a per-step update loses most of its digits — doing it in log space
    does not.
    """
    log_post = trace.log_posterior
    # Not a shape test: at n_contexts == 2 the joint posterior is (B, T, R, R) and
    # the naive one is (B, T, C, R) — both 4-D, and indistinguishable by ndim.
    if trace.name == "naive":  # already one axis per variable
        return log_post - _logsumexp(log_post, axis=-1)
    marginals = []
    for c in range(n_contexts):
        other = tuple(2 + k for k in range(n_contexts) if k != c)
        marginal = (
            _logsumexp(log_post, axis=other).squeeze(other) if other else log_post
        )
        marginals.append(marginal - _logsumexp(marginal, axis=-1))
    return np.stack(marginals, axis=2)


def _normalized_update(log_belief: np.ndarray) -> np.ndarray:
    """Per-step multiplicative update to a belief, renormalized; time is axis 1.

    Bayes makes each step a multiplication, so differencing consecutive *log*
    beliefs recovers the factor that was applied. Step 0 has no predecessor and
    keeps the belief itself.
    """
    log_update = np.empty_like(log_belief)
    log_update[:, 0] = log_belief[:, 0]
    log_update[:, 1:] = log_belief[:, 1:] - log_belief[:, :-1]
    return np.exp(log_update - _logsumexp(log_update, axis=-1))


def disentanglement(
    joint: BeliefTrace,
    naive: BeliefTrace,
    batch: EpisodeBatch,
    *,
    per_variable: bool = False,
) -> np.ndarray:
    """How much of each step's belief update was carried by the history.

    Returns ``(n_episodes, n_steps)`` for the goal variable, or
    ``(n_episodes, n_steps, n_contexts)`` with ``per_variable=True``.

    The Jeffreys divergence (arXiv:2603.27134 §B.3) between the naive marginal
    likelihood ``p_Z(o_t | r)`` and the history-conditioned one
    ``p_Z(o_t | r, o_1:t-1)``, the latter recovered from the ratio of consecutive
    optimal marginal beliefs. The two coincide exactly when the marginal belief
    dynamics are Markovian, so any divergence is entanglement between the goal
    and the context variables.

    Jeffreys is the *sum* of both KL directions, ``D_KL(p||q) + D_KL(q||p)``, not
    their average.

    Unlike :func:`factorization_regret` this is a per-step quantity, not an
    accumulating one — a flow rather than a stock. Neither bounds the other: KL
    is not additive across sequential Bayesian updates.
    """
    n_contexts = batch.cfg.n_contexts
    conditioned = _normalized_update(_log_marginal_belief(joint, n_contexts))
    factorized = _normalized_update(_log_marginal_belief(naive, n_contexts))
    per_var = (
        (conditioned - factorized)
        * (np.log(conditioned) - np.log(factorized))
    ).sum(-1)
    if per_variable:
        return per_var
    return per_var[np.arange(batch.n_episodes), :, batch.goal_ind]
