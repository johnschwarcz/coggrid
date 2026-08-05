"""The generative model, as pure functions.

Every function here takes explicit arguments and an explicit ``rng``, and
returns arrays. Nothing reads or writes ``self``, nothing caches, nothing plots.
That is the whole point: each stage can be tested, swapped or called in
isolation.

The pipeline, in order:

.. code-block:: text

    orthonormal_embeddings   ->  K, Q                 (n_vars, n_obs, dim)
    sample_contexts          ->  ctx_inds, goal_ind   (n_eps, n_ctx)
    value_profile            ->  profile              (n_realizations, n_roll)
    joint_likelihood         ->  rates, interactions  (n_eps, n_obs, R, ..., R)
    marginal_likelihood      ->  factorized rates     (n_eps, n_obs, n_ctx, R)
    sample_realizations      ->  ctx_vals, goal_value (n_eps, n_ctx)
    observation_rates        ->  p(obs=1)             (n_eps, n_obs)
    sample_observations      ->  observations         (n_eps, n_steps, n_obs)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .config import CogGridConfig

__all__ = [
    "Split",
    "orthonormal_embeddings",
    "value_profile",
    "realization_potential",
    "interaction_strength",
    "joint_likelihood",
    "marginal_likelihood",
    "sample_contexts",
    "sample_realizations",
    "observation_rates",
    "sample_observations",
    "sigmoid",
]

Split = Literal["train", "held_out"]


# --------------------------------------------------------------------------- #
# numerics
# --------------------------------------------------------------------------- #
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic function.

    Branching on the sign avoids the ``inf`` intermediates (and overflow
    warning) that ``1 / (1 + np.exp(-x))`` produces for strongly negative ``x``.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


# --------------------------------------------------------------------------- #
# 1. variable embeddings
# --------------------------------------------------------------------------- #
def orthonormal_embeddings(
    n_vars: int,
    n_observations: int,
    embedding_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random unit vectors, orthogonalized across observation channels.

    Returns an array of shape ``(n_vars, n_observations, embedding_dim)`` in
    which, for every variable ``s``, the ``n_observations`` vectors
    ``out[s, :, :]`` form an orthonormal set.

    Orthogonality across channels is what makes the observation channels carry
    *independent* information about the same pair of latent variables: the
    interaction strength seen by channel ``o`` is uncorrelated with the one seen
    by channel ``o'``.

    This requires ``embedding_dim >= n_observations``; :class:`CogGridConfig`
    enforces it.
    """
    if embedding_dim < n_observations:
        raise ValueError(
            f"embedding_dim ({embedding_dim}) < n_observations ({n_observations})"
        )
    v = rng.standard_normal((n_vars, n_observations, embedding_dim))
    for o in range(n_observations):
        # Modified Gram-Schmidt against the already-normalized earlier channels.
        for prev in range(o):
            projection = np.einsum("sk,sk->s", v[:, o], v[:, prev])
            v[:, o] -= projection[:, None] * v[:, prev]
        v[:, o] /= np.linalg.norm(v[:, o], axis=-1, keepdims=True)
    return v


# --------------------------------------------------------------------------- #
# 2. the circular value profile
# --------------------------------------------------------------------------- #
def value_profile(cfg: CogGridConfig) -> np.ndarray:
    """Rolled sinusoidal value profile, shape ``(n_realizations, n_roll)``.

    Row ``r`` is the base sinusoid circularly shifted by ``r``. Interaction
    strengths index into this profile (see :func:`realization_potential`), so
    the shift is what makes each realization of a latent variable prefer a
    different interaction strength.

    ``n_roll = 1 + 2 * n_realizations`` is deliberately longer than
    ``n_realizations``: it gives the smoothing kernel room to wrap around
    instead of piling mass up at the edges of the realization axis.
    """
    grid = np.linspace(-1.0, 1.0, cfg.n_roll + 1)[1:]
    base = cfg.likelihood_temp * np.sin(grid * np.pi * cfg.likelihood_freq)
    return np.stack([np.roll(base, r) for r in range(cfg.n_realizations)])


def realization_potential(
    strength: np.ndarray, profile: np.ndarray, n_roll: int
) -> np.ndarray:
    """Map interaction strengths to a potential over realizations.

    Parameters
    ----------
    strength:
        Array of any shape ``(...)`` of inner products between a key and a query
        embedding.
    profile:
        ``(n_realizations, n_roll)`` array from :func:`value_profile`.
    n_roll:
        Length of the profile.

    Returns
    -------
    ``(..., n_realizations)`` potential.

    Notes
    -----
    A strength ``z`` selects a position ``-z * n_roll`` on a circle of
    ``n_roll`` slots. A Gaussian kernel over *circular* distance turns that into
    a normalized weight vector, which is then dotted against each rolled row of
    the profile. Smooth, periodic, and monotone in ``|z|`` locally — which is
    what lets nearby latent variables produce nearby observation statistics.
    """
    offsets = np.arange(n_roll)
    shifted = (offsets - np.asarray(strength)[..., None] * n_roll) % n_roll
    distance = np.minimum(shifted, n_roll - shifted)
    weights = np.exp(-(distance**2))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ profile.T


# --------------------------------------------------------------------------- #
# 3. the joint likelihood
# --------------------------------------------------------------------------- #
def interaction_strength(
    active_keys: np.ndarray, active_queries: np.ndarray, i: int, j: int
) -> np.ndarray:
    """Inner product between the key of active variable ``i`` and the query of ``j``.

    ``active_keys`` / ``active_queries`` have shape
    ``(n_episodes, n_contexts, n_observations, embedding_dim)``; the result has
    shape ``(n_episodes, n_observations)``.

    Note this is *asymmetric*: ``(i, j)`` and ``(j, i)`` give different numbers,
    which is what lets a pair of variables produce an anisotropic joint
    distribution rather than a symmetric one.
    """
    return np.einsum(
        "bod,bod->bo", active_keys[:, i], active_queries[:, j]
    )


def joint_likelihood(
    cfg: CogGridConfig,
    keys: np.ndarray,
    queries: np.ndarray,
    ctx_inds: np.ndarray,
    profile: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bernoulli rates for every joint realization of the active variables.

    Parameters
    ----------
    cfg:
        The world configuration.
    keys, queries:
        ``(n_vars, n_observations, embedding_dim)`` embeddings.
    ctx_inds:
        ``(n_episodes, n_contexts)`` indices of the active latent variables.
    profile:
        Optional precomputed :func:`value_profile` output.

    Returns
    -------
    rates:
        ``(n_episodes, n_observations, n_realizations, ..., n_realizations)``
        with ``n_contexts`` trailing realization axes. ``rates[b, o, r0, r1]``
        is ``P(observation o == 1)`` in episode ``b`` when active variable 0 takes
        value ``r0`` and active variable 1 takes value ``r1``.
    interactions:
        ``(n_episodes, n_observations, k)`` raw strengths, one per ordered pair
        of active variables in the order ``(0,1), (1,0), (0,2), (2,0), ...``.
        Useful for analysis; not needed to run the environment.

    Notes
    -----
    The rate is ``sigmoid`` of a sum of **pairwise** potentials. There is no
    higher-order term, so the log-odds are pairwise-decomposable — but the
    *probability* is not, and neither is the resulting observation
    distribution. That non-factorization is the thing this environment exists
    to measure.
    """
    if profile is None:
        profile = value_profile(cfg)

    ctx_inds = np.asarray(ctx_inds)
    n_episodes, n_contexts = ctx_inds.shape
    if n_contexts != cfg.n_contexts:
        raise ValueError(
            f"ctx_inds has {n_contexts} contexts but cfg.n_contexts={cfg.n_contexts}"
        )

    active_keys = keys[ctx_inds]        # (n_eps, n_ctx, n_obs, dim)
    active_queries = queries[ctx_inds]

    logits = np.zeros(cfg.joint_likelihood_shape(n_episodes))
    strengths: list[np.ndarray] = []

    if n_contexts == 1:
        # Degenerate case: a variable interacts with itself, giving a single
        # potential over the one realization axis.
        z = interaction_strength(active_keys, active_queries, 0, 0)
        logits += realization_potential(z, profile, cfg.n_roll)
        strengths.append(z)
    else:
        for i in range(n_contexts):
            for j in range(i + 1, n_contexts):
                z_ij = interaction_strength(active_keys, active_queries, i, j)
                z_ji = interaction_strength(active_keys, active_queries, j, i)
                v_i = realization_potential(z_ij, profile, cfg.n_roll)
                v_j = realization_potential(z_ji, profile, cfg.n_roll)

                # Outer product over the two realization axes, then broadcast
                # into the full (n_ctx)-dimensional realization grid by giving
                # every other context a length-1 axis.
                pair = np.einsum("bom,bon->bomn", v_i, v_j)
                broadcast_shape = [1] * n_contexts
                broadcast_shape[i] = cfg.n_realizations
                broadcast_shape[j] = cfg.n_realizations
                logits += pair.reshape(
                    n_episodes, cfg.n_observations, *broadcast_shape
                )
                strengths.extend([z_ij, z_ji])

    return sigmoid(logits), np.stack(strengths, axis=-1)


def marginal_likelihood(rates: np.ndarray, n_contexts: int) -> np.ndarray:
    """Collapse the joint rates to one independent table per active variable.

    Returns ``(n_episodes, n_observations, n_contexts, n_realizations)``, where
    entry ``[b, o, c, r]`` averages the joint rate over all realizations of
    every context *other* than ``c``.

    This is the likelihood available to an observer that has (wrongly) assumed
    the active variables are independent. Averaging — rather than maximizing or
    conditioning — corresponds to marginalizing under a uniform prior over the
    other variables' values, which is exactly the prior an agent starts an episode
    with.
    """
    rates = np.asarray(rates)
    n_episodes, n_observations = rates.shape[:2]
    n_realizations = rates.shape[2]
    out = np.empty((n_episodes, n_observations, n_contexts, n_realizations))
    for c in range(n_contexts):
        other_axes = tuple(2 + i for i in range(n_contexts) if i != c)
        out[:, :, c, :] = rates.mean(axis=other_axes) if other_axes else rates
    return out


# --------------------------------------------------------------------------- #
# 4. sampling contexts, values and observations
# --------------------------------------------------------------------------- #
def _draw(
    pool: np.ndarray,
    n_episodes: int,
    n_contexts: int,
    distinct: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``(n_episodes, n_contexts)`` variables from ``pool``."""
    if not distinct:
        return rng.choice(pool, size=(n_episodes, n_contexts))
    if n_contexts > pool.size:
        raise ValueError(
            f"cannot draw {n_contexts} distinct variables from a pool of {pool.size}"
        )
    # Vectorized per-row sampling without replacement.
    order = np.argsort(rng.random((n_episodes, pool.size)), axis=1)
    return pool[order[:, :n_contexts]]


def sample_contexts(
    cfg: CogGridConfig,
    rng: np.random.Generator,
    split: Split = "held_out",
    n_episodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose which latent variables are active, and which one carries the goal.

    Parameters
    ----------
    split:
        ``"held_out"``
            Every active variable comes from the held-out pool
            (``range(n_held_out_vars)``). Nothing in the episode was seen
            during training — the hard generalization condition.
        ``"train"``
            Variables come from the whole pool, but **at most one** held-out variable
            is allowed per episode and the goal is never a held-out variable. This
            is the compositional condition: a novel variable appears alongside
            familiar ones, and the agent is scored on a familiar one.

    Returns
    -------
    ctx_inds:
        ``(n_episodes, n_contexts)`` sorted indices of the active variables.
    goal_ind:
        ``(n_episodes,)`` which *column* of ``ctx_inds`` is the goal.

    Notes
    -----
    With ``n_contexts == 1`` the ``"train"`` split degenerates: there is no
    second variable to pair a novel one with, so the "at most one held-out variable"
    rule is vacuous and the single variable (hence the goal) may be held out.
    """
    n_episodes = cfg.n_episodes if n_episodes is None else n_episodes
    all_vars = np.arange(cfg.n_vars)
    n_held_out = int(cfg.n_held_out_vars or 0)
    held_out, train = all_vars[:n_held_out], all_vars[n_held_out:]

    if cfg.subsample_vars is not None:
        held_out = rng.choice(held_out, size=cfg.subsample_vars, replace=False)
        train = rng.choice(train, size=cfg.subsample_vars, replace=False)

    distinct = not cfg.allow_repeated_vars

    if split == "held_out":
        ctx = _draw(held_out, n_episodes, cfg.n_contexts, distinct, rng)
        ctx_inds = np.sort(ctx, axis=1)
        goal_ind = rng.random((n_episodes, cfg.n_contexts)).argmax(axis=1)
        return ctx_inds, goal_ind

    if split != "train":
        raise ValueError(f"split must be 'train' or 'held_out', got {split!r}")

    pool = (
        all_vars
        if cfg.subsample_vars is None
        else np.concatenate([train, held_out])
    )
    ctx = _draw(pool, n_episodes, cfg.n_contexts, distinct, rng)

    # Keep at most one held-out variable per episode; replace any extras.
    is_held_out = np.isin(ctx, held_out)
    keep = np.zeros_like(is_held_out)
    keep[np.arange(n_episodes), is_held_out.argmax(axis=1)] = is_held_out.any(axis=1)
    surplus = is_held_out & ~keep
    if surplus.any():
        ctx[surplus] = rng.choice(train, size=int(surplus.sum()))
    ctx_inds = np.sort(ctx, axis=1)

    # The goal must be a familiar variable: zero out held-out columns before argmax.
    priority = rng.random((n_episodes, cfg.n_contexts))
    priority[np.isin(ctx_inds, held_out)] = 0.0
    return ctx_inds, priority.argmax(axis=1)


def sample_realizations(
    cfg: CogGridConfig,
    rng: np.random.Generator,
    goal_ind: np.ndarray,
    n_episodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a value for each active variable, uniformly over realizations.

    Returns ``(ctx_vals, goal_value)`` with shapes ``(n_episodes, n_contexts)``
    and ``(n_episodes,)``.
    """
    n_episodes = cfg.n_episodes if n_episodes is None else n_episodes
    ctx_vals = rng.integers(0, cfg.n_realizations, size=(n_episodes, cfg.n_contexts))
    goal_value = ctx_vals[np.arange(n_episodes), goal_ind]
    return ctx_vals, goal_value


def observation_rates(rates: np.ndarray, ctx_vals: np.ndarray) -> np.ndarray:
    """Pick out the true Bernoulli rate for the realization that actually occurred.

    ``rates`` is the full joint table; ``ctx_vals`` says which cell is real.
    Returns ``(n_episodes, n_observations)``.
    """
    rates = np.asarray(rates)
    ctx_vals = np.asarray(ctx_vals)
    n_episodes, n_observations = rates.shape[:2]
    index = (
        np.arange(n_episodes)[:, None],
        np.arange(n_observations)[None, :],
        *(ctx_vals[:, c, None] for c in range(ctx_vals.shape[1])),
    )
    return rates[index]


def sample_observations(
    true_rates: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n_steps`` i.i.d. Bernoulli observation vectors per episode.

    Returns a boolean ``(n_episodes, n_steps, n_observations)`` array.

    The rate does not change over the episode — this is what "stationary" means
    here. All the difficulty is in the *ambiguity* of the mapping from
    realizations to rates, not in tracking a moving target.
    """
    true_rates = np.asarray(true_rates)
    n_episodes, n_observations = true_rates.shape
    draws = rng.random((n_episodes, n_steps, n_observations))
    return true_rates[:, None, :] > draws
