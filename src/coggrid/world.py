"""``World`` (a configured environment) and ``EpisodeBatch`` (what it samples).

* :class:`World` holds what is fixed for a whole experiment: the config and the variable embeddings.
* :class:`EpisodeBatch` is a plain, immutable record of a batch of complete episodes.
  The observers, the gym environment and the figures all consume it.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import likelihood, sampling
from .config import CogGridConfig
from .sampling import Split

__all__ = ["World", "EpisodeBatch", "EmbeddingSource"]

#: ``(cfg, rng) -> (keys, queries)``, each of shape ``(n_vars, n_observations, embedding_dim)``.
EmbeddingSource = Callable[[CogGridConfig, np.random.Generator], tuple[np.ndarray, np.ndarray]]

def _npz(path: str | Path) -> Path:
    """``path`` with ``.npz`` appended unless it already ends so, as numpy names the file."""
    path = Path(path)
    return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")


def _default_embeddings(cfg: CogGridConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    keys = likelihood.orthonormal_embeddings(cfg.n_vars, cfg.n_observations, cfg.embedding_dim, rng)
    queries = likelihood.orthonormal_embeddings(cfg.n_vars, cfg.n_observations, cfg.embedding_dim, rng)
    return keys, queries

class World:
    """A CogGrid world: the config plus the variable embeddings, drawn once and shared by every batch, which is what makes the train/held-out split meaningful.
    ``rng`` draws the embeddings.
    ``embeddings`` replaces the random orthonormal embeddings with your own :data:`EmbeddingSource`.

    >>> world = World(CogGridConfig(n_vars=60, n_contexts=2, seed=0))
    >>> world.sample_episodes(8).observations.shape   # (batch_size, n_steps, n_observations)
    (8, 30, 5)
    """

    def __init__(self, cfg: CogGridConfig | None = None, *, rng: int | np.random.Generator | None = None, embeddings: EmbeddingSource | None = None) -> None:
        self.cfg = cfg if cfg is not None else CogGridConfig()
        self.rng = self.cfg.rng(rng)
        self.keys, self.queries = (embeddings or _default_embeddings)(self.cfg, self.rng)
        expected = (self.cfg.n_vars, self.cfg.n_observations, self.cfg.embedding_dim)
        for name, arr in (("keys", self.keys), ("queries", self.queries)):
            if arr.shape != expected:
                raise ValueError(f"embedding source returned {name} with shape {arr.shape}, expected {expected}")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        cfg = self.cfg
        return f"World(n_vars={cfg.n_vars}, n_contexts={cfg.n_contexts}, n_realizations={cfg.n_realizations}, n_observations={cfg.n_observations})"

    def sample_episodes(self, batch_size: int | None = None, *, split: Split = "held_out", rng: int | np.random.Generator | None = None, expose_likelihood: bool = True) -> EpisodeBatch:
        """Draw a batch of complete episodes.

        ``split`` is ``"held_out"`` (every variable testing) or ``"train"`` (at most one testing variable, never the goal unless ``n_contexts=1``);
        see :func:`~coggrid.sampling.sample_contexts`.
        ``expose_likelihood=False`` skips the likelihood tables, ``O(n_realizations ** n_contexts)``, making ``joint_likelihood`` and ``marginal_likelihood`` equal to ``None``.
        """
        cfg = self.cfg
        n = cfg.batch_size if batch_size is None else int(batch_size)
        if n < 1:
            raise ValueError(f"batch_size must be >= 1, got {n}")
        if expose_likelihood:
            cfg.warn_if_large(n)
        batch_rng = self.rng if rng is None else cfg.rng(rng)

        ctx_inds, goal_ind = sampling.sample_contexts(cfg, batch_rng, split, n)
        ctx_vals, goal_val = sampling.sample_realizations(cfg, batch_rng, goal_ind, n)
        # The likelihood draws nothing from the rng, so it leaves the stream as it is.
        p_obs__vals, interactions = likelihood.p_obs__vals(cfg, self.keys, self.queries, ctx_inds, ctx_vals)
        joint: np.ndarray | None = None
        marginal: np.ndarray | None = None
        if expose_likelihood:
            joint = likelihood.joint_likelihood(cfg, interactions)
            marginal = likelihood.marginal_likelihood(joint)
        observations = sampling.sample_observations(p_obs__vals, cfg.n_steps, batch_rng)

        return EpisodeBatch(
            cfg=cfg,
            split=split,
            ctx_inds=ctx_inds,
            goal_ind=goal_ind,
            ctx_vals=ctx_vals,
            goal_val=goal_val,
            joint_likelihood=joint,
            marginal_likelihood=marginal,
            p_obs__vals=p_obs__vals,
            observations=observations,
            interactions=interactions,
        )

@dataclass(frozen=True, slots=True)
class EpisodeBatch:
    """A batch of episodes. Every array has a leading ``batch_size`` axis.
    Shapes use ``R`` for ``n_realizations`` and ``k`` for the number of ordered pairs of active variables, ``n_contexts * (n_contexts - 1)`` (1 with a single context).

    Attributes
    ----------
    cfg: The config these episodes were drawn from.
    split: Which variable pool the contexts came from.
    ctx_inds: ``(batch_size, n_contexts)`` indices of the active latent variables, sorted.
    goal_ind: ``(batch_size,)`` which column of ``ctx_inds`` the agent is scored on.
    ctx_vals: ``(batch_size, n_contexts)`` the realization each active variable took.
    goal_val: ``(batch_size,)`` the goal variable's realization, the quantity to infer.
    joint_likelihood: ``(batch_size, n_observations, R, ..., R)`` ``P(obs = 1 | r)`` for every joint realization ``r``, ``None`` if sampled with ``expose_likelihood=False``.
    marginal_likelihood: ``(batch_size, n_observations, n_contexts, R)`` the same, averaged over the other variables, ``None`` if sampled with ``expose_likelihood=False``.
    p_obs__vals: ``(batch_size, n_observations)`` ``P(obs = 1 | ctx_vals)``, which generated the observations: ``joint_likelihood`` at ``ctx_vals``.
    observations: ``(batch_size, n_steps, n_observations)`` the boolean observation stream.
    interactions: ``(batch_size, n_observations, k)`` the interaction phase of each ordered pair of active variables.
    """

    cfg: CogGridConfig
    split: Split
    ctx_inds: np.ndarray
    goal_ind: np.ndarray
    ctx_vals: np.ndarray
    goal_val: np.ndarray
    joint_likelihood: np.ndarray | None
    marginal_likelihood: np.ndarray | None
    p_obs__vals: np.ndarray
    observations: np.ndarray
    interactions: np.ndarray

    @property
    def batch_size(self) -> int:
        return int(self.ctx_inds.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.observations.shape[1])

    def __len__(self) -> int:
        return self.batch_size

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        cfg = self.cfg
        return (
            f"EpisodeBatch(batch_size={self.batch_size}, split={self.split!r}, n_contexts={cfg.n_contexts}, "
            f"n_realizations={cfg.n_realizations}, n_observations={cfg.n_observations}, n_steps={self.n_steps})"
        )

    def _tables_of(self, index: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(joint_likelihood, marginal_likelihood)`` of the selected episodes, rebuilt if absent."""
        part = self.select(index)
        joint = part.joint_likelihood
        if joint is None:
            joint = likelihood.joint_likelihood(self.cfg, part.interactions)
        return joint, likelihood.marginal_likelihood(joint)

    def select(self, index: Any) -> EpisodeBatch:
        """A new batch with only the selected episodes.

        ``index`` may be an int, slice, boolean mask or integer array; an int gives a batch of size 1.
        """
        if isinstance(index, (int, np.integer)):
            index = np.array([index])

        def part(table: np.ndarray | None) -> np.ndarray | None:
            return None if table is None else table[index]

        return replace(
            self,
            ctx_inds=self.ctx_inds[index],
            goal_ind=self.goal_ind[index],
            ctx_vals=self.ctx_vals[index],
            goal_val=self.goal_val[index],
            joint_likelihood=part(self.joint_likelihood),
            marginal_likelihood=part(self.marginal_likelihood),
            p_obs__vals=self.p_obs__vals[index],
            observations=self.observations[index],
            interactions=self.interactions[index],
        )

    def save(self, path: str | Path) -> Path:
        """Write to a compressed ``.npz``, config included."""
        path = _npz(path)
        tables = {"joint_likelihood": self.joint_likelihood, "marginal_likelihood": self.marginal_likelihood}
        np.savez_compressed(
            path,
            _config=np.array([repr(asdict(self.cfg))]),
            _split=np.array([self.split]),
            ctx_inds=self.ctx_inds,
            goal_ind=self.goal_ind,
            ctx_vals=self.ctx_vals,
            goal_val=self.goal_val,
            p_obs__vals=self.p_obs__vals,
            observations=self.observations,
            interactions=self.interactions,
            **{name: table for name, table in tables.items() if table is not None},
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> EpisodeBatch:
        """Read a batch written by :meth:`save`, from the same ``path``."""
        with np.load(_npz(path), allow_pickle=False) as data:

            def table(name: str) -> np.ndarray | None:
                return data[name] if name in data.files else None

            return cls(
                cfg=CogGridConfig(**ast.literal_eval(str(data["_config"][0]))),
                split=str(data["_split"][0]),  # type: ignore[arg-type]
                ctx_inds=data["ctx_inds"],
                goal_ind=data["goal_ind"],
                ctx_vals=data["ctx_vals"],
                goal_val=data["goal_val"],
                joint_likelihood=table("joint_likelihood"),
                marginal_likelihood=table("marginal_likelihood"),
                p_obs__vals=data["p_obs__vals"],
                observations=data["observations"],
                interactions=data["interactions"],
            )

