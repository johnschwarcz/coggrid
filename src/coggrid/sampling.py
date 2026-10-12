"""The random draws of an episode: which variables are active, the realizations they take, and the observations they produce.

:meth:`World.sample_episodes <coggrid.World.sample_episodes>` makes them in this order,
and between the last two, :func:`coggrid.likelihood.p_obs__vals` turns the realizations into ``P(obs = 1 | ctx_vals)``:

    sample_contexts      ->  ctx_inds, goal_ind  (batch_size, n_ctx), (batch_size,)
    sample_realizations  ->  ctx_vals, goal_val  (batch_size, n_ctx), (batch_size,)
    sample_observations  ->  observations        (batch_size, n_steps, n_obs)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .config import CogGridConfig

__all__ = ["Split", "sample_contexts", "sample_realizations", "sample_observations"]

Split = Literal["train", "held_out"]


def sample_contexts(cfg: CogGridConfig, rng: np.random.Generator, split: Split = "held_out", batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Choose the active variables, ``ctx_inds`` ``(batch_size, n_contexts)`` sorted, and which column is the goal, ``goal_ind`` ``(batch_size,)``.

    ``"held_out"`` draws every variable from the held-out pool ``range(n_held_out_vars)``.
    ``"train"`` draws from all variables but keeps at most one held-out variable per episode and never makes it the goal:
    a novel variable appears alongside familiar ones.
    With one context there is no split: the lone variable is the goal, so ``"train"`` draws from every variable.

    Unless ``cfg.allow_repeated_vars``, an episode never activates a variable twice:
    two slots holding one variable cannot be told apart, so a goal among them could not be identified.
    Such episodes are drawn again (:func:`_redraw_repeats`); with a given seed, all others are unchanged.
    """
    batch_size = cfg.batch_size if batch_size is None else batch_size
    all_vars = np.arange(cfg.n_vars)
    n_held_out = int(cfg.n_held_out_vars or 0)
    held_out, train = all_vars[:n_held_out], all_vars[n_held_out:]

    if split == "held_out":
        ctx = rng.choice(held_out, size=(batch_size, cfg.n_contexts))
        if not cfg.allow_repeated_vars:
            _redraw_repeats(ctx, split, held_out, train, rng)
        ctx_inds = np.sort(ctx, axis=1)
        goal_ind = rng.random((batch_size, cfg.n_contexts)).argmax(axis=1)
        return ctx_inds, goal_ind

    if split != "train":
        raise ValueError(f"split must be 'train' or 'held_out', got {split!r}")
    ctx = rng.choice(all_vars, size=(batch_size, cfg.n_contexts))

    # Keep at most one held-out variable per episode; replace any extras.
    is_held_out = np.isin(ctx, held_out)
    keep = np.zeros_like(is_held_out)
    keep[np.arange(batch_size), is_held_out.argmax(axis=1)] = is_held_out.any(axis=1)
    surplus = is_held_out & ~keep
    if surplus.any():
        ctx[surplus] = rng.choice(train, size=int(surplus.sum()))
    if not cfg.allow_repeated_vars:
        _redraw_repeats(ctx, split, held_out, train, rng)
    ctx_inds = np.sort(ctx, axis=1)

    # The goal must be a familiar variable: zero out held-out columns before argmax.
    priority = rng.random((batch_size, cfg.n_contexts))
    priority[np.isin(ctx_inds, held_out)] = 0.0
    return ctx_inds, priority.argmax(axis=1)


def _redraw_repeats(ctx: np.ndarray, split: Split, held_out: np.ndarray, train: np.ndarray, rng: np.random.Generator) -> None:
    """Redraw, in place, the rows of ``ctx`` that activate a variable twice.

    They are drawn from the law all rows follow, given distinct variables, so the batch is exactly the repeating sampler conditioned on no repeats.
    The draws come from a side stream (a jumped copy of ``rng``), so all other rows, and everything drawn after, comes out as with ``allow_repeated_vars=True``.
    A generator that cannot jump (SFC64) seeds the side stream from ``rng`` instead, which shifts every later draw.
    """
    in_order = np.sort(ctx, axis=1)
    rows = np.flatnonzero((in_order[:, 1:] == in_order[:, :-1]).any(axis=1))
    if not rows.size:
        return
    jumped = getattr(rng.bit_generator, "jumped", None)
    side = np.random.default_rng(rng.integers(2**63)) if jumped is None else np.random.Generator(jumped())
    n_ctx = ctx.shape[1]

    def subsets(pool: np.ndarray, n: int, size: int) -> np.ndarray:
        return pool[side.random((n, pool.size)).argsort(axis=1)[:, :size]]

    if split == "held_out":
        ctx[rows] = subsets(held_out, rows.size, n_ctx)
        return
    # Training rows hold one held-out variable with probability 1 - q**C, q the pool's training share, and independent training draws
    # otherwise. Given that those differ, one held-out variable is likelier by the ratio of the chances that C - 1 and C training draws all differ.
    q = train.size / (train.size + held_out.size)
    one, none = 1 - q**n_ctx, q**n_ctx * (1 - (n_ctx - 1) / train.size)
    with_one = side.random(rows.size) < one / (one + none)
    k = int(with_one.sum())
    redrawn = np.empty((rows.size, n_ctx), dtype=ctx.dtype)
    redrawn[with_one, 0] = side.choice(held_out, size=k)
    redrawn[with_one, 1:] = subsets(train, k, n_ctx - 1)
    redrawn[~with_one] = subsets(train, rows.size - k, n_ctx)
    ctx[rows] = redrawn


def sample_realizations(cfg: CogGridConfig, rng: np.random.Generator, goal_ind: np.ndarray, batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A uniform realization for each active variable: ``ctx_vals`` ``(batch_size, n_contexts)`` and ``goal_val`` ``(batch_size,)``."""
    batch_size = cfg.batch_size if batch_size is None else batch_size
    ctx_vals = rng.integers(0, cfg.n_realizations, size=(batch_size, cfg.n_contexts))
    goal_val = ctx_vals[np.arange(batch_size), goal_ind]
    return ctx_vals, goal_val


def sample_observations(p_obs__vals: np.ndarray, n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """``(batch_size, n_steps, n_observations)`` boolean observations: channel ``o`` reads 1 with probability ``p_obs__vals[:, o]``.

    They are i.i.d. over steps: the world is stationary, and all the difficulty is in inferring the values behind them.
    """
    p_obs__vals = np.asarray(p_obs__vals)
    batch_size, n_observations = p_obs__vals.shape
    draws = rng.random((batch_size, n_steps, n_observations))
    return p_obs__vals[:, None, :] > draws
