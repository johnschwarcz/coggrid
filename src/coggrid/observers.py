"""Ideal-observer baselines and the metrics used to score them.

* ``joint``: exact Bayes over the full joint realization of all active variables, using :attr:`EpisodeBatch.joint_likelihood`. The Bayes-optimal observer.
* ``naive``: one independent posterior per active variable, using :attr:`EpisodeBatch.marginal_likelihood`.
  Exactly the joint observer with the interactions averaged away, so the gap between the two isolates the cost of assuming a factorized latent space.

Inference runs in log space on running counts of the evidence, so it does not underflow at long horizons.
The joint observer works a few episodes at a time, rebuilding each chunk's table if the batch was sampled with ``expose_likelihood=False``;
``device=`` runs it with PyTorch instead (see :mod:`coggrid.torch`).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from . import likelihood
from .world import EpisodeBatch

__all__ = ["BeliefTrace", "joint_observer", "naive_observer", "run_observers", "factorization_regret", "disentanglement", "score_belief"]

ObserverName = Literal["joint", "naive"]

#: Likelihoods are clipped into ``[LIKELIHOOD_FLOOR, 1 - LIKELIHOOD_FLOOR]`` before taking logs.
#: This binds only when the sigmoid saturates: with a large ``likelihood_temp``, or with many contexts.
LIKELIHOOD_FLOOR = 1e-12


@dataclass(frozen=True, slots=True)
class BeliefTrace:
    """What an observer believed, and how good that was, at every timestep.

    Attributes
    ----------
    name: ``"joint"`` or ``"naive"``.
    belief: ``(batch_size, n_steps, n_contexts, n_realizations)`` marginal posterior over each active variable's value.
    goal_belief: ``(batch_size, n_steps, n_realizations)`` the goal variable's slice of ``belief``.
    estimate: ``(batch_size, n_steps, n_contexts)`` posterior mean realization.
    accuracy: ``(batch_size, n_steps)`` 1.0 where the posterior mode equals the true goal value.
    p_correct: ``(batch_size, n_steps)`` posterior probability of the true goal value, a softer, lower-variance accuracy.
    mse: ``(batch_size, n_steps)`` squared error of the posterior mean.
    log_posterior: ``(batch_size, n_steps, ...)`` unnormalized log posterior, which is the cumulative log-likelihood since the prior is uniform.
        Over the joint realization grid for the joint observer, and over ``(n_contexts, n_realizations)`` for the naive one.
        ``None`` if the joint observer ran with ``keep_log_posterior=False``.
    log_goal_belief: ``(batch_size, n_steps, n_realizations)`` normalized log posterior over the goal variable's value, which :func:`factorization_regret` reads while it matches ``goal_belief``.
        Finite and exact where ``goal_belief`` underflows to 0, since the observers marginalize it in log space there. ``None`` on a trace built by hand.
        Swap another belief into a trace (with :func:`dataclasses.replace` and :func:`score_belief`, say) and this no longer describes it: set it to ``None`` too.
    """

    name: ObserverName
    belief: np.ndarray
    goal_belief: np.ndarray
    estimate: np.ndarray
    accuracy: np.ndarray
    p_correct: np.ndarray
    mse: np.ndarray
    log_posterior: np.ndarray | None
    log_goal_belief: np.ndarray | None = None

    def mean_curves(self) -> dict[str, np.ndarray]:
        """Episode-averaged learning curves, each ``(n_steps,)``."""
        return {"accuracy": self.accuracy.mean(0), "p_correct": self.p_correct.mean(0), "mse": self.mse.mean(0)}

    def final(self) -> dict[str, float]:
        """Scalar summary at the last timestep."""
        return {"accuracy": float(self.accuracy[:, -1].mean()), "p_correct": float(self.p_correct[:, -1].mean()), "mse": float(self.mse[:, -1].mean())}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        f = self.final()
        return f"BeliefTrace({self.name!r}, batch_size={self.belief.shape[0]}, final acc={f['accuracy']:.3f}, p_correct={f['p_correct']:.3f})"


def _logsumexp(x: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    peak = np.max(x, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return peak + np.log(np.exp(x - peak).sum(axis=axis, keepdims=True))


def _cumulative_log_likelihood(observations: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Bernoulli log-likelihood of the evidence so far:
    ``(batch_size, n_steps, n_obs)`` observations and a ``(batch_size, n_obs, *hypotheses)`` likelihood of a 1 give ``(batch_size, n_steps, *hypotheses)``.

    Observations are i.i.d. given the latent values, so after ``t`` steps the log-likelihood depends only on how often each channel has fired:
    one matrix product of those counts against ``[log p; log(1 - p)]``.
    """
    p = np.clip(np.asarray(table, dtype=float), LIKELIHOOD_FLOOR, 1.0 - LIKELIHOOD_FLOOR)
    ones = np.cumsum(observations, axis=1, dtype=float)
    seen = np.arange(1, ones.shape[1] + 1)[:, None]
    counts = np.concatenate([ones, seen - ones], axis=-1)
    log_p = np.concatenate([np.log(p), np.log1p(-p)], axis=1)
    out = counts @ log_p.reshape(len(p), log_p.shape[1], -1)
    return out.reshape(out.shape[:2] + p.shape[2:])


def _joint_table(batch: EpisodeBatch, eps: slice) -> np.ndarray:
    """The joint likelihood of episodes ``eps``, rebuilt if the batch has none."""
    if batch.joint_likelihood is not None:
        return batch.joint_likelihood[eps]
    return likelihood.joint_likelihood(batch.cfg, batch.interactions[eps])


def _chunks(batch: EpisodeBatch) -> list[slice]:
    """Slices of episodes holding about ``likelihood._CHUNK_CELLS`` numbers of log posterior each."""
    n = max(1, likelihood._CHUNK_CELLS // (batch.n_steps * math.prod(batch.cfg.realization_shape)))
    return [slice(lo, lo + n) for lo in range(0, batch.batch_size, n)]


def _log_goal(belief: np.ndarray, goal_ind: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The log of each episode's goal belief, ``(n, n_steps, R)``, and which episodes' beliefs come near underflow.
    The log is exact while the belief stays far above float64's smallest normal number, 2.2e-308;
    the episodes that come closer, at long horizons, need their goal marginal summed out in log space instead, by :func:`_log_goal_marginal`."""
    goal = belief[np.arange(len(goal_ind)), :, goal_ind]
    with np.errstate(divide="ignore"):
        return np.log(goal), np.flatnonzero(goal.min(axis=(1, 2)) < 1e-250)


def _log_goal_marginal(log_post: np.ndarray, goal_ind: np.ndarray) -> np.ndarray:
    """``(n, n_steps, R, ..., R)`` joint log posterior to ``(n, n_steps, R)``, each episode's normalized log marginal over its goal variable.
    Episodes are grouped by which axis is their goal and the others summed out in log space, so tails that underflow as probabilities stay finite.
    With one goal axis among them it works in ``log_post``, overwriting it, to hold no second copy."""
    out = np.empty(log_post.shape[:3])
    for c in np.unique(goal_ind):
        rows = np.flatnonzero(goal_ind == c)
        other = tuple(a for a in range(2, log_post.ndim) if a != 2 + c)
        marginal = log_post if rows.size == len(log_post) else log_post[rows]
        if other:
            peak = marginal.max(axis=other, keepdims=True)
            marginal -= peak
            # A term below e^-700 of the peak cannot move a sum of at least 1, and clamping it keeps np.exp off its slow path for subnormals.
            np.maximum(marginal, -700.0, out=marginal)
            marginal = (peak + np.log(np.exp(marginal, out=marginal).sum(axis=other, keepdims=True))).squeeze(other)
        out[rows] = marginal - _logsumexp(marginal, axis=-1)
    return out


def joint_observer(batch: EpisodeBatch, *, keep_log_posterior: bool = True, device: Any = None) -> BeliefTrace:
    """Exact Bayesian inference over all ``n_realizations ** n_contexts`` joint realizations, marginalized to per-variable beliefs.
    ``keep_log_posterior=True`` keeps the ``(batch_size, n_steps, R, ..., R)`` log posterior, which :func:`disentanglement` needs.
    ``device`` (e.g. ``"cuda"``) runs on a PyTorch device instead.
    """
    if device is not None:
        from .torch import _joint_arrays

        belief, log_posterior, log_goal_belief = _joint_arrays(batch, device, keep_log_posterior)
        return _finish("joint", belief, log_posterior, log_goal_belief, batch)

    n_contexts = batch.cfg.n_contexts
    realization_axes = tuple(range(2, 2 + n_contexts))
    belief = np.empty((batch.batch_size, batch.n_steps, n_contexts, batch.cfg.n_realizations))
    log_goal_belief = np.empty((batch.batch_size, batch.n_steps, batch.cfg.n_realizations))
    shape = (batch.batch_size, batch.n_steps, *batch.cfg.realization_shape)
    log_posterior = np.empty(shape) if keep_log_posterior else None

    for eps in _chunks(batch):
        table = _joint_table(batch, eps)
        log_post = _cumulative_log_likelihood(batch.observations[eps], table)
        if log_posterior is not None:
            log_posterior[eps] = log_post
        log_post -= log_post.max(axis=realization_axes, keepdims=True)
        rest = np.exp(log_post, out=log_post).reshape(*log_post.shape[:2], -1)
        # Marginalize the joint posterior to one distribution per variable, peeling one off at a time: two full passes in all.
        for c in range(n_contexts - 1):
            rest = rest.reshape(*rest.shape[:2], batch.cfg.n_realizations, -1)
            belief[eps, :, c, :] = rest.sum(axis=-1)
            rest = rest.sum(axis=2)
        belief[eps, :, -1, :] = rest
        belief[eps] /= belief[eps, :, :1].sum(axis=-1, keepdims=True)
        goal_ind = batch.goal_ind[eps]
        log_goal_belief[eps], deep = _log_goal(belief[eps], goal_ind)
        if deep.size:
            # Their log posterior was exponentiated in place above, so it is rebuilt, once the chunk is freed: one is held at a time.
            del log_post, rest
            log_goal_belief[eps.start + deep] = _log_goal_marginal(_cumulative_log_likelihood(batch.observations[eps][deep], table[deep]), goal_ind[deep])

    return _finish("joint", belief, log_posterior, log_goal_belief, batch)


def naive_observer(batch: EpisodeBatch) -> BeliefTrace:
    """Bayesian inference under the (false) assumption of independent variables: the marginal likelihood, each variable's posterior normalized separately.
    Its shortfall against :func:`joint_observer` is the price of a factorized representation."""
    marginal = batch.marginal_likelihood
    if marginal is None:  # expose_likelihood=False: from each chunk's rebuilt table
        marginal = np.concatenate([likelihood.marginal_likelihood(_joint_table(batch, eps)) for eps in _chunks(batch)])
    log_posterior = _cumulative_log_likelihood(batch.observations, marginal)
    log_belief = log_posterior - _logsumexp(log_posterior, axis=-1)
    log_goal_belief = log_belief[np.arange(batch.batch_size), :, batch.goal_ind]
    return _finish("naive", np.exp(log_belief, out=log_belief), log_posterior, log_goal_belief, batch)


def _finish(name: ObserverName, belief: np.ndarray, log_posterior: np.ndarray | None, log_goal_belief: np.ndarray, batch: EpisodeBatch) -> BeliefTrace:
    return BeliefTrace(name=name, belief=belief, log_posterior=log_posterior, log_goal_belief=log_goal_belief, **score_belief(belief, batch))


def run_observers(batch: EpisodeBatch, *, keep_log_posterior: bool = True, device: Any = None) -> dict[ObserverName, BeliefTrace]:
    """Run both baselines: ``{"joint": ..., "naive": ...}``.
    The keyword arguments go to :func:`joint_observer`; the naive observer always runs in numpy, rebuilding the tables on the CPU if the batch has none."""
    joint = joint_observer(batch, keep_log_posterior=keep_log_posterior, device=device)
    return {"joint": joint, "naive": naive_observer(batch)}


def score_belief(belief: np.ndarray, batch: EpisodeBatch) -> dict[str, np.ndarray]:
    """Score any ``(batch_size, n_steps, n_contexts, n_realizations)`` belief against the truth in ``batch``, a learned agent's as well as an observer's.
    Returns ``goal_belief``, ``estimate``, ``accuracy``, ``p_correct`` and ``mse``;
    ``dataclasses.replace(trace, belief=belief, log_goal_belief=None, **score_belief(belief, batch))`` puts the belief in an observer's place."""
    belief = np.asarray(belief, dtype=float)
    expected_tail = (batch.cfg.n_contexts, batch.cfg.n_realizations)
    if belief.ndim != 4 or belief.shape[2:] != expected_tail:
        raise ValueError(f"belief must have shape (batch_size, n_steps, {expected_tail[0]}, {expected_tail[1]}), got {belief.shape}")

    episodes = np.arange(belief.shape[0])
    estimate = belief @ np.arange(batch.cfg.n_realizations)    # (eps, steps, ctx)
    goal_belief = belief[episodes, :, batch.goal_ind, :]       # (eps, steps, real)
    truth = batch.goal_val[:, None]                            # (eps, 1)
    accuracy = (goal_belief.argmax(axis=-1) == truth).astype(float)
    p_correct = np.take_along_axis(goal_belief, truth[..., None], axis=-1).squeeze(-1)
    mse = (truth - estimate[episodes, :, batch.goal_ind]) ** 2
    return {"goal_belief": goal_belief, "estimate": estimate, "accuracy": accuracy, "p_correct": p_correct, "mse": mse}


def factorization_regret(joint: BeliefTrace, naive: BeliefTrace, *, symmetric: bool = False, clip: float | None = None, eps: float | None = None) -> np.ndarray:
    """``(batch_size, n_steps)`` divergence of the naive posterior over the goal from the joint one: large where modelling the interaction mattered.

    The default is the paper's definition (arXiv:2603.27134), the forward ``D_KL(B_joint || B_naive)``; ``symmetric=True`` averages both directions.
    It is exact by default, computed from each trace's ``log_goal_belief``: finite and correct where the beliefs themselves underflow to 0, and uncapped, so values above 34.5 nats are real.
    A trace without one, or whose ``goal_belief`` it no longer matches (checked at the last step, where a belief swapped in shows), is read from ``goal_belief`` instead,
    with exact zeros floored at the smallest positive float64.
    ``clip`` clips both beliefs into ``[clip, 1 - clip]`` first, which caps each term at ``-ln(clip)``:
    ``clip=1e-15`` (34.5 nats) reproduces the paper's regret curves, which saturate there, and coggrid <= 0.3.0. ``eps`` is a deprecated alias of ``clip``.
    """
    name = "clip"
    if eps is not None:
        if clip is not None:
            raise TypeError("pass clip, or its deprecated alias eps, not both")
        warnings.warn("factorization_regret's eps is deprecated: use clip, or leave it out for the exact divergence", DeprecationWarning, stacklevel=2)
        clip, name = eps, "eps"
    if clip is not None:
        if not 0.0 < clip < 0.5:
            raise ValueError(f"{name} must be in (0, 0.5), or left out for the exact divergence, got {clip}")
        p = np.clip(joint.goal_belief, clip, 1.0 - clip)
        q = np.clip(naive.goal_belief, clip, 1.0 - clip)
        forward = (p * np.log(p / q)).sum(-1)
        if not symmetric:
            return forward
        reverse = (q * np.log(q / p)).sum(-1)
        return (forward + reverse) / 2.0

    log_p, log_q = _log_goal_belief(joint), _log_goal_belief(naive)
    forward = _kl(log_p, log_q)
    if not symmetric:
        return forward
    return (forward + _kl(log_q, log_p)) / 2.0


def _log_goal_belief(trace: BeliefTrace) -> np.ndarray:
    """A trace's log belief over the goal, in float64: its ``log_goal_belief`` where that describes its ``goal_belief``,
    else the log of ``goal_belief``, with exact zeros floored at the smallest positive float64."""
    goal = np.asarray(trace.goal_belief, dtype=np.float64)
    log_goal = getattr(trace, "log_goal_belief", None)  # unset in a trace pickled by coggrid <= 0.3.0
    if log_goal is not None:
        log_goal = np.asarray(log_goal, dtype=np.float64)
        # Beyond rounding, and the precision probabilities lose near underflow, they differ only if another belief was swapped into the trace.
        # That shows at the last step, which is all that is compared: the whole array would cost as much as the divergence itself.
        if log_goal.shape == goal.shape and np.allclose(np.exp(log_goal[:, -1]), goal[:, -1], rtol=1e-6, atol=1e-250):
            return log_goal
    return np.log(np.maximum(goal, np.finfo(np.float64).tiny))


def _kl(log_p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
    """``sum_r p(r) [log p(r) - log q(r)]`` over the last axis, from log beliefs.
    Terms with ``p(r) < e^-700`` (1e-304) are dropped as too small to count, ``p(r) = 0`` among them: 0 is that term's limit, even where ``log q(r) = -inf`` makes the product NaN.
    Clamped there, the exponential also stays off numpy's slow path for subnormals."""
    p = np.maximum(log_p, -700.0)
    np.exp(p, out=p)
    with np.errstate(invalid="ignore"):
        return np.where(log_p > -700.0, p * (log_p - log_q), 0.0).sum(-1)


def _log_marginal_belief(trace: BeliefTrace, n_contexts: int) -> np.ndarray:
    """``(batch_size, n_steps, n_contexts, n_realizations)`` log marginal posterior, read off the stored log posterior:
    beliefs fall below 1e-12 once an observer is confident, and per-step updates divided out of them would lose most of their digits."""
    log_post = trace.log_posterior
    if log_post is None:
        raise ValueError("this needs the joint observer's keep_log_posterior=True")
    # By name, not shape: at n_contexts == 2 the joint (B, T, R, R) and naive (B, T, C, R) posteriors are both 4-D.
    if trace.name == "naive":
        return log_post - _logsumexp(log_post, axis=-1)
    marginals = []
    for c in range(n_contexts):
        other = tuple(2 + k for k in range(n_contexts) if k != c)
        marginal = _logsumexp(log_post, axis=other).squeeze(other) if other else log_post
        marginals.append(marginal - _logsumexp(marginal, axis=-1))
    return np.stack(marginals, axis=2)


def _normalized_update(log_belief: np.ndarray) -> np.ndarray:
    """Per-step multiplicative update to a belief, renormalized (time is axis 1): differencing consecutive log beliefs recovers the factor Bayes applied."""
    log_update = np.empty_like(log_belief)
    log_update[:, 0] = log_belief[:, 0]
    log_update[:, 1:] = log_belief[:, 1:] - log_belief[:, :-1]
    return np.exp(log_update - _logsumexp(log_update, axis=-1))


def disentanglement(joint: BeliefTrace, naive: BeliefTrace, batch: EpisodeBatch, *, per_variable: bool = False) -> np.ndarray:
    """How much of each step's belief update was carried by the history:
    ``(batch_size, n_steps)`` for the goal, or ``(batch_size, n_steps, n_contexts)`` with ``per_variable=True``.
    The Jeffreys divergence (the sum of both KL directions; arXiv:2603.27134)
    between the naive marginal likelihood ``p(o_t | r)`` and the history-conditioned one ``p(o_t | r, o_1:t-1)``, recovered from consecutive optimal marginal beliefs.
    They coincide exactly when the marginal belief dynamics are Markovian, so any divergence is entanglement between the goal and the other active variables.
    A per-step flow, unlike the accumulating :func:`factorization_regret`.
    """
    n_contexts = batch.cfg.n_contexts
    conditioned = _normalized_update(_log_marginal_belief(joint, n_contexts))
    factorized = _normalized_update(_log_marginal_belief(naive, n_contexts))
    per_var = ((conditioned - factorized) * (np.log(conditioned) - np.log(factorized))).sum(-1)
    if per_variable:
        return per_var
    return per_var[np.arange(batch.batch_size), :, batch.goal_ind]
