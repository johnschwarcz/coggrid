"""``World`` (a configured environment) and ``EpisodeBatch`` (its output).

The original design put all of this on one object: instantiating
``CognitiveGridworld`` preprocessed the environment, generated embeddings,
generated episodes, ran the Bayesian observers, *and* drew plots, all from
``__init__``. Results were left scattered across ~40 attributes on ``self``.

Here the split is:

* :class:`World` owns the things that are fixed for a whole experiment — the
  config and the state embeddings. Building one is cheap and side-effect free.
* :class:`EpisodeBatch` is a plain, immutable record of one batch of episodes.
  It is the only thing observers, environments and plotting functions consume.

Nothing in this module runs an observer or opens a figure.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import generative as gen
from .components import (
    ContextSampler,
    EmbeddingSource,
    LikelihoodModel,
    ObservationModel,
    RealizationSampler,
)
from .config import CogGridConfig
from .generative import Split

__all__ = ["World", "EpisodeBatch"]


# --------------------------------------------------------------------------- #
# default component implementations
# --------------------------------------------------------------------------- #
def _default_embeddings(
    cfg: CogGridConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    keys = gen.orthonormal_embeddings(
        cfg.n_states, cfg.n_observations, cfg.embedding_dim, rng
    )
    queries = gen.orthonormal_embeddings(
        cfg.n_states, cfg.n_observations, cfg.embedding_dim, rng
    )
    return keys, queries


def _default_likelihood(
    cfg: CogGridConfig,
    keys: np.ndarray,
    queries: np.ndarray,
    ctx_inds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return gen.joint_likelihood(cfg, keys, queries, ctx_inds)


def _default_observations(
    cfg: CogGridConfig,
    rates: np.ndarray,
    ctx_vals: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    true_rates = gen.observation_rates(rates, ctx_vals)
    observations = gen.sample_observations(true_rates, cfg.n_steps, rng)
    return true_rates, observations


# --------------------------------------------------------------------------- #
# EpisodeBatch
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class EpisodeBatch:
    """One batch of episodes, fully specified.

    Everything an observer, an agent wrapper or a plotting function needs, and
    nothing else. All arrays share a leading ``n_episodes`` axis.

    Attributes
    ----------
    cfg:
        The config these episodes were drawn from.
    split:
        Which state pool the contexts came from.
    ctx_inds:
        ``(n_episodes, n_contexts)`` indices of the active latent states, sorted.
    goal_ind:
        ``(n_episodes,)`` which column of ``ctx_inds`` the agent is scored on.
    ctx_vals:
        ``(n_episodes, n_contexts)`` the realisation each active state took.
    goal_value:
        ``(n_episodes,)`` the realisation of the goal state — the quantity to
        be inferred.
    rates:
        ``(n_episodes, n_observations, *(n_realizations,) * n_contexts)``
        Bernoulli rates for every hypothetical joint realisation. This is the
        likelihood function an ideal observer has access to.
    marginal_rates:
        ``(n_episodes, n_observations, n_contexts, n_realizations)`` the same
        thing after wrongly assuming the active states are independent.
    true_rates:
        ``(n_episodes, n_observations)`` the rate that actually generated the
        observations, i.e. ``rates`` indexed at ``ctx_vals``.
    observations:
        ``(n_episodes, n_steps, n_observations)`` boolean observation stream.
    interactions:
        ``(n_episodes, n_observations, n_interactions)`` raw interaction
        strengths, kept for analysis.
    """

    cfg: CogGridConfig
    split: Split
    ctx_inds: np.ndarray
    goal_ind: np.ndarray
    ctx_vals: np.ndarray
    goal_value: np.ndarray
    rates: np.ndarray
    marginal_rates: np.ndarray
    true_rates: np.ndarray
    observations: np.ndarray
    interactions: np.ndarray

    # ------------------------------------------------------------------ shape
    @property
    def n_episodes(self) -> int:
        return int(self.ctx_inds.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.observations.shape[1])

    def __len__(self) -> int:
        return self.n_episodes

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"EpisodeBatch(n_episodes={self.n_episodes}, split={self.split!r}, "
            f"n_contexts={self.cfg.n_contexts}, "
            f"n_realizations={self.cfg.n_realizations}, "
            f"n_observations={self.cfg.n_observations}, n_steps={self.n_steps})"
        )

    # --------------------------------------------------------------- indexing
    def select(self, index: Any) -> "EpisodeBatch":
        """Return a new batch containing only the selected episodes.

        ``index`` may be an int, slice, boolean mask or integer array. An int
        yields a batch of size 1 rather than dropping the axis, so downstream
        code never has to special-case it.
        """
        if isinstance(index, (int, np.integer)):
            index = np.array([index])
        return replace(
            self,
            ctx_inds=self.ctx_inds[index],
            goal_ind=self.goal_ind[index],
            ctx_vals=self.ctx_vals[index],
            goal_value=self.goal_value[index],
            rates=self.rates[index],
            marginal_rates=self.marginal_rates[index],
            true_rates=self.true_rates[index],
            observations=self.observations[index],
            interactions=self.interactions[index],
        )

    # ------------------------------------------------------------------- i/o
    def save(self, path: str | Path) -> Path:
        """Write to a compressed ``.npz``, config included."""
        from dataclasses import asdict

        path = Path(path).with_suffix(".npz")
        np.savez_compressed(
            path,
            _config=np.array([repr(asdict(self.cfg))]),
            _split=np.array([self.split]),
            ctx_inds=self.ctx_inds,
            goal_ind=self.goal_ind,
            ctx_vals=self.ctx_vals,
            goal_value=self.goal_value,
            rates=self.rates,
            marginal_rates=self.marginal_rates,
            true_rates=self.true_rates,
            observations=self.observations,
            interactions=self.interactions,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EpisodeBatch":
        """Read a batch written by :meth:`save`."""
        import ast

        with np.load(Path(path), allow_pickle=False) as data:
            cfg = CogGridConfig(**ast.literal_eval(str(data["_config"][0])))
            return cls(
                cfg=cfg,
                split=str(data["_split"][0]),  # type: ignore[arg-type]
                ctx_inds=data["ctx_inds"],
                goal_ind=data["goal_ind"],
                ctx_vals=data["ctx_vals"],
                goal_value=data["goal_value"],
                rates=data["rates"],
                marginal_rates=data["marginal_rates"],
                true_rates=data["true_rates"],
                observations=data["observations"],
                interactions=data["interactions"],
            )


# --------------------------------------------------------------------------- #
# World
# --------------------------------------------------------------------------- #
class World:
    """A configured CogGrid world: state embeddings plus a sampler.

    Building a ``World`` draws the state embeddings once. Sampling episodes from
    it is then cheap and repeatable, and the same embeddings are shared across
    every batch — which is what makes the train/held-out split meaningful.

    Parameters
    ----------
    cfg:
        The configuration. Defaults to ``CogGridConfig()``.
    rng:
        Seed or ``Generator`` used to draw the embeddings. Falls back to
        ``cfg.seed``.
    embeddings, contexts, likelihood, realizations, observations:
        Optional replacements for the five generative stages. See
        :mod:`coggrid.components`.

    Examples
    --------
    >>> from coggrid import CogGridConfig, World
    >>> world = World(CogGridConfig(n_states=60, n_contexts=2, seed=0))
    >>> batch = world.sample_episodes(8)
    >>> batch.observations.shape
    (8, 30, 5)
    """

    def __init__(
        self,
        cfg: CogGridConfig | None = None,
        *,
        rng: int | np.random.Generator | None = None,
        embeddings: EmbeddingSource | None = None,
        contexts: ContextSampler | None = None,
        likelihood: LikelihoodModel | None = None,
        realizations: RealizationSampler | None = None,
        observations: ObservationModel | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else CogGridConfig()
        self.rng = self.cfg.rng(rng)

        self._embeddings: EmbeddingSource = embeddings or _default_embeddings
        self._contexts: ContextSampler = contexts or gen.sample_contexts
        self._likelihood: LikelihoodModel = likelihood or _default_likelihood
        self._realizations: RealizationSampler = realizations or gen.sample_realizations
        self._observations: ObservationModel = observations or _default_observations

        self.keys, self.queries = self._embeddings(self.cfg, self.rng)
        self._check_embeddings()

        #: Cached value profile — a deterministic function of the config.
        self.profile = gen.value_profile(self.cfg)

    def _check_embeddings(self) -> None:
        expected = (self.cfg.n_states, self.cfg.n_observations, self.cfg.embedding_dim)
        for name, arr in (("keys", self.keys), ("queries", self.queries)):
            if arr.shape != expected:
                raise ValueError(
                    f"embedding source returned {name} with shape {arr.shape}, "
                    f"expected {expected}"
                )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"World(n_states={self.cfg.n_states}, n_contexts={self.cfg.n_contexts}, "
            f"n_realizations={self.cfg.n_realizations}, "
            f"n_observations={self.cfg.n_observations})"
        )

    # ---------------------------------------------------------------- sampling
    def sample_episodes(
        self,
        n_episodes: int | None = None,
        *,
        split: Split = "held_out",
        rng: int | np.random.Generator | None = None,
    ) -> EpisodeBatch:
        """Draw a batch of episodes.

        Parameters
        ----------
        n_episodes:
            Batch size. Defaults to ``cfg.n_episodes``.
        split:
            ``"held_out"`` (all active states novel) or ``"train"`` (at most one
            novel state, goal always familiar). See
            :func:`~coggrid.generative.sample_contexts`.
        rng:
            Optional seed or ``Generator`` for *this batch only*. Passing an int
            here makes the batch exactly reproducible without rebuilding the
            world.
        """
        cfg = self.cfg
        n = cfg.n_episodes if n_episodes is None else int(n_episodes)
        if n < 1:
            raise ValueError(f"n_episodes must be >= 1, got {n}")
        cfg.warn_if_large(n)

        batch_rng = self.rng if rng is None else cfg.rng(rng)

        ctx_inds, goal_ind = self._contexts(cfg, batch_rng, split, n)
        rates, interactions = self._likelihood(cfg, self.keys, self.queries, ctx_inds)
        marginal_rates = gen.marginal_likelihood(rates, cfg.n_contexts)
        ctx_vals, goal_value = self._realizations(cfg, batch_rng, goal_ind, n)
        true_rates, observations = self._observations(cfg, rates, ctx_vals, batch_rng)

        return EpisodeBatch(
            cfg=cfg,
            split=split,
            ctx_inds=ctx_inds,
            goal_ind=goal_ind,
            ctx_vals=ctx_vals,
            goal_value=goal_value,
            rates=rates,
            marginal_rates=marginal_rates,
            true_rates=true_rates,
            observations=observations,
            interactions=interactions,
        )
