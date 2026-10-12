"""Likelihood: maps active variables' realizations to observation probabilities: ``p_obs__vals`` = ``P(obs = 1 | r = ctx_vals)``.
keys and queries computed when :class:`~coggrid.World` is built, p_obs__vals computed every :meth:`~coggrid.World.sample_episodes`.

    orthonormal_embeddings  ->  keys, queries                                 (n_vars, n_obs, dim)
    get_sinusoids           ->  sinusoids                                     (n_realizations, n_roll)
    p_obs__vals             ->  p_obs__vals, interactions                     (batch_size, n_obs), (batch_size, n_obs, k)
    joint_likelihood        ->  P(obs = 1 | r) for every joint realization r  (batch_size, n_obs, R, ..., R)
    marginal_likelihood     ->  the same, averaged over the other variables   (batch_size, n_obs, n_ctx, R)
``R`` is ``n_realizations`` and ``k`` the number of ordered pairs.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from .config import CogGridConfig

__all__ = [
    "orthonormal_embeddings",
    "get_sinusoids",
    "soft_roll",
    "p_obs__vals",
    "joint_likelihood",
    "marginal_likelihood",
    "sigmoid",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic function: ``exp(-|x|)`` never overflows."""
    x = np.asarray(x, dtype=float)
    e = np.exp(np.fmin(x, -x))  # fmin rather than -abs passes NaN through unchanged
    # out= keeps a 0-d input a 0-d array rather than a numpy scalar.
    return np.divide(np.where(x >= 0, 1.0, e), 1.0 + e, out=np.empty_like(x))


# ----------------------------------------------------------------------- embeddings
def orthonormal_embeddings(n_vars: int, n_observations: int, embedding_dim: int, rng: np.random.Generator) -> np.ndarray:
    """``(n_vars, n_observations, embedding_dim)`` random unit vectors.
    Each variable's are orthogonal across channels, so each channel carries independent information."""
    if embedding_dim < n_observations:
        raise ValueError(f"embedding_dim ({embedding_dim}) < n_observations ({n_observations})")
    v = rng.standard_normal((n_vars, n_observations, embedding_dim))
    for o in range(n_observations):  # modified Gram-Schmidt against the earlier channels
        for prev in range(o):
            projection = np.einsum("sk,sk->s", v[:, o], v[:, prev])
            v[:, o] -= projection[:, None] * v[:, prev]
        v[:, o] /= np.linalg.norm(v[:, o], axis=-1, keepdims=True)
    return v


# ---------------------------------------------------------------------- the likelihood
def get_sinusoids(cfg: CogGridConfig) -> np.ndarray:
    """``(n_realizations, n_roll)``: row ``r`` is a sinusoid circularly shifted by ``r``.
    ``n_roll = 1 + 2 * n_realizations`` leaves the smoothing kernel room to wrap around instead of piling up at the edges.
    """
    grid = np.linspace(-1.0, 1.0, cfg.n_roll + 1)[1:]
    base = cfg.likelihood_temp * np.sin(grid * np.pi * cfg.likelihood_freq)
    return np.stack([np.roll(base, r) for r in range(cfg.n_realizations)])


def soft_roll(phase: np.ndarray, sinusoids: np.ndarray) -> np.ndarray:
    """Shift the sinusoid by each interaction ``phase`` ``(...)`` and read it at every realization: one outer vector ``(..., n_realizations)`` per phase.

    A phase ``z`` picks a position ``z * n_roll`` on a circle of ``n_roll`` slots, the length of each sinusoid.
    A Gaussian kernel over circular distance turns it into weights over those slots, a soft (fractional) roll, and each realization's row of ``sinusoids`` is averaged with them.
    """
    n_roll = sinusoids.shape[-1]
    shifted = (np.arange(n_roll) - np.asarray(phase)[..., None] * n_roll) % n_roll
    distance = np.minimum(shifted, n_roll - shifted)
    weights = np.exp(-(distance**2))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ sinusoids.T


def p_obs__vals(cfg: CogGridConfig, keys: np.ndarray, queries: np.ndarray, ctx_inds: np.ndarray, ctx_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``P(obs = 1 | ctx_vals)``, ``(batch_size, n_observations)``: the probabilities that generate the observations, and the ``interactions`` behind them.
    ``interactions``, ``(batch_size, n_observations, k)``, is the phase of each ordered pair of active variables, in the order ``(0,1), (1,0), (0,2), (2,0), ...``.
    """
    ctx_inds = np.asarray(ctx_inds)
    if ctx_inds.shape[1] != cfg.n_contexts:
        raise ValueError(f"ctx_inds has {ctx_inds.shape[1]} contexts but cfg.n_contexts={cfg.n_contexts}")
    sinusoids = get_sinusoids(cfg)
    # Pairs per chunk: each holds n_obs x (sinusoid length or embedding width) numbers.
    chunk = max(1, _CHUNK_CELLS // (cfg.n_observations * max(cfg.n_roll, cfg.embedding_dim)))
    phases, index = _distinct_phases(keys, queries, ctx_inds, chunk)
    # Each distinct pair's outer vector, laid out (pair, realization, channel).
    outer_vecs = np.empty((phases.shape[0], cfg.n_realizations, phases.shape[1]))
    for lo in range(0, phases.shape[0], chunk):
        outer_vecs[lo : lo + chunk] = soft_roll(phases[lo : lo + chunk], sinusoids).transpose(0, 2, 1)

    def at_realization(slot: int, c: int) -> np.ndarray:
        return outer_vecs[index[:, slot], ctx_vals[:, c]]

    logits = np.zeros((len(ctx_inds), cfg.n_observations))
    _add_pairwise(logits, cfg.n_contexts, at_realization)
    interactions = np.ascontiguousarray(phases[index].transpose(0, 2, 1))
    return sigmoid(logits), interactions


def joint_likelihood(cfg: CogGridConfig, interactions: np.ndarray) -> np.ndarray:
    """``P(obs = 1 | r)`` for every joint realization ``r`` of the active variables, from the ``interactions`` of :func:`p_obs__vals`.
    table, ``(batch_size, n_observations, R, ..., R)``, has one realization axis per context.
    index ``[b, o, r0, r1]`` is ``P(obs = 1 | r0, r1)`` on channel ``o`` of episode ``b``.
    """
    sinusoids = get_sinusoids(cfg)
    batch_size = interactions.shape[0]
    table = np.empty(cfg.joint_likelihood_shape(batch_size))
    step = max(1, _CHUNK_CELLS // math.prod(table.shape[1:]))
    for lo in range(0, batch_size, step):
        # (n, k, n_obs, R). Phases stay stacked (..., n_obs): numpy's matmul rounds differently depending on the stack shape.
        outer_vecs = soft_roll(interactions[lo : lo + step].transpose(0, 2, 1), sinusoids)
        n = outer_vecs.shape[0]

        def on_grid(slot: int, c: int, outer_vecs=outer_vecs, n=n) -> np.ndarray:
            # Context c's outer vector along its own axis of the realization grid.
            shape = [1] * cfg.n_contexts
            shape[c] = cfg.n_realizations
            return outer_vecs[:, slot].reshape(n, cfg.n_observations, *shape)

        logits = np.zeros((n, *table.shape[1:]))
        _add_pairwise(logits, cfg.n_contexts, on_grid)
        table[lo : lo + n] = sigmoid(logits)
    return table


def marginal_likelihood(joint: np.ndarray) -> np.ndarray:
    """``(batch_size, n_observations, n_contexts, R)``: the joint likelihood averaged over every other context's value."""
    joint = np.asarray(joint)
    axes = range(2, joint.ndim)  # one realization axis per context
    marginals = np.stack([joint.mean(axis=tuple(other for other in axes if other != keep)) for keep in axes], axis=2)
    return marginals.astype(float, order="C", copy=False)  # float64 and C-ordered, whatever the joint's dtype and layout


#: Big arrays are built and reduced in pieces of about this many numbers (512 KiB of float64), so temporaries stay small and in cache.
#: It changes only speed and peak memory, never results.
_CHUNK_CELLS = 1 << 16


def _distinct_phases(keys: np.ndarray, queries: np.ndarray, ctx_inds: np.ndarray, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """``phases``, ``(n_distinct, n_obs)``: one variable's key dotted with the other's query, once per distinct ordered pair.

    ``index``, ``(batch_size, k)``: the row of ``phases`` for each pair of each episode.
    The same pairs recur across a batch, so with many contexts this saves most of the work.
    """
    n_vars = keys.shape[0]
    ctx_inds = np.arange(n_vars, dtype=np.int64)[ctx_inds]  # int64, so pair ids cannot overflow
    n_ctx = ctx_inds.shape[1]
    # Key-side and query-side context of each ordered pair, in the order (0,1), (1,0), (0,2), (2,0), ..., or the self-pair (0,0) alone.
    ordered = [(a, b) for i in range(n_ctx) for j in range(i + 1, n_ctx) for a, b in ((i, j), (j, i))] or [(0, 0)]
    key_side, query_side = np.array(ordered, dtype=np.intp).T
    pair_id = ctx_inds[:, key_side] * n_vars + ctx_inds[:, query_side]
    distinct, index = np.unique(pair_id, return_inverse=True)
    phases = np.empty((distinct.size, keys.shape[1]), dtype=np.result_type(keys, queries))
    for lo in range(0, distinct.size, chunk):
        pairs = distinct[lo : lo + chunk]
        phases[lo : lo + chunk] = np.einsum("bod,bod->bo", keys[pairs // n_vars], queries[pairs % n_vars])
    return phases, index.reshape(pair_id.shape)


def _add_pairwise(logits: np.ndarray, n_contexts: int, place: Callable[[int, int], np.ndarray]) -> None:
    """Add every pair's outer product into ``logits``, in order.
    ``place(slot, c)`` is context ``c``'s outer vector in pair slot ``slot``: spread over the grid for the table, or read at the realized values for ``p_obs__vals``.
    Both share this loop, so the table read at ``ctx_vals`` equals ``p_obs__vals`` to the last bit."""
    if n_contexts == 1:  # a lone variable interacts with itself: one outer vector over its realizations
        logits += place(0, 0)
        return
    slot = 0
    for i in range(n_contexts):
        for j in range(i + 1, n_contexts):
            logits += place(slot, i) * place(slot + 1, j)
            slot += 2
