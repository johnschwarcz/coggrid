"""Every tunable of a CogGrid world, in one validated, immutable dataclass."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

__all__ = ["CogGridConfig"]


@dataclass(frozen=True, slots=True)
class CogGridConfig:
    """Immutable description of a CogGrid world, a stationary POMDP. On each episode:

    1. ``n_contexts`` latent variables are drawn from a pool of ``n_vars``;
    2. The likelihood comes from pairwise interactions between the active variables.
    3. each active variable takes one of ``n_realizations`` discrete values;
    4. those values jointly set the likelihood, the probability that each of ``n_observations`` binary channels reads 1;
    5. the agent sees ``n_steps`` i.i.d. samples of those channels and must infer the value of the single *goal* variable.

    Attributes
    ----------
    n_vars: Size of the latent variable pool.
    n_contexts: Active variables per episode. The joint likelihood table is ``n_realizations ** n_contexts`` wide, so this drives memory.
    n_realizations: Values each active variable can take.
    n_observations: Binary observation channels.
    n_steps: Observation samples per episode (the horizon).
    embedding_dim: Length of each key/query embedding; at least ``n_observations``, since embeddings are orthogonalized across channels.
    likelihood_temp: Amplitude of the sinusoids; larger pushes the likelihood towards 0/1.
    likelihood_freq: Number of periods in each sinusoid.
    batch_size: Episodes per :meth:`~coggrid.World.sample_episodes` call (and per refill of :class:`~coggrid.CogGridEnv`) when no number is given.
    n_held_out_vars: Size of the held-out pool, ``range(n_held_out_vars)``; the rest are training variables. ``None`` means ``n_vars // 10``.
    allow_repeated_vars: Whether one episode may activate the same variable twice.  Off by default: two slots holding one variable cannot be told apart.
    seed: Seed for the default RNG. ``None`` means non-reproducible.
    """

    n_vars: int = 500
    n_contexts: int = 2
    n_realizations: int = 10
    n_observations: int = 5
    n_steps: int = 30
    embedding_dim: int = 30
    likelihood_temp: float = 2.0
    likelihood_freq: float = 1.0
    batch_size: int = 1000
    n_held_out_vars: int | None = None
    allow_repeated_vars: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        for name in ("n_vars", "n_contexts", "n_realizations", "n_observations", "n_steps", "embedding_dim", "batch_size"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.embedding_dim < self.n_observations:
            raise ValueError(f"embedding_dim ({self.embedding_dim}) must be >= n_observations ({self.n_observations}).")
        if self.likelihood_temp <= 0:
            raise ValueError(f"likelihood_temp must be > 0, got {self.likelihood_temp}")

        # Resolve the "auto" default once, so every reader downstream sees an int.
        held_out = self.n_vars // 10 if self.n_held_out_vars is None else self.n_held_out_vars
        object.__setattr__(self, "n_held_out_vars", held_out)
        if not 0 <= held_out <= self.n_vars:
            raise ValueError(f"n_held_out_vars ({held_out}) must be in [0, n_vars={self.n_vars}]")
        if held_out == 0:
            raise ValueError("n_held_out_vars resolved to 0.")
        if held_out == self.n_vars:
            raise ValueError("n_held_out_vars equals n_vars.")
        smaller_pool = min(held_out, self.n_vars - held_out)
        if not self.allow_repeated_vars and self.n_contexts > smaller_pool:
            pool = "held-out" if held_out <= self.n_vars - held_out else "training"
            raise ValueError(f"n_contexts ({self.n_contexts}) exceeds the {pool} pool ({smaller_pool} variables). Enlarge it, or set allow_repeated_vars=True.")

    @property
    def n_train_vars(self) -> int:
        """Number of latent variables reserved for training episodes."""
        return self.n_vars - self.n_held_out_vars

    @property
    def n_roll(self) -> int:
        """Length of each sinusoid, in slots around a circle: ``1 + 2 * n_realizations``."""
        return 1 + 2 * self.n_realizations

    @property
    def realization_shape(self) -> tuple[int, ...]:
        """Shape of the joint realization axes: ``(n_realizations,) * n_contexts``."""
        return (self.n_realizations,) * self.n_contexts

    def joint_likelihood_shape(self, batch_size: int | None = None) -> tuple[int, ...]:
        """Shape of the joint likelihood table for a batch."""
        n = self.batch_size if batch_size is None else batch_size
        return (n, self.n_observations, *self.realization_shape)

    def memory_report(self, batch_size: int | None = None) -> str:
        """Human-readable estimate of the dominant allocation."""
        n = self.batch_size if batch_size is None else batch_size
        table = int(np.prod(self.joint_likelihood_shape(n), dtype=np.int64)) * 8
        log_posterior = table * self.n_steps / self.n_observations

        def human(x: float) -> str:
            for unit in ("B", "KiB", "MiB", "GiB"):
                if x < 1024:
                    return f"{x:.1f} {unit}"
                x /= 1024
            return f"{x:.1f} TiB"

        return (
            f"{n} episodes x {self.n_observations} channels x {self.n_realizations}^{self.n_contexts} realizations\n"
            f"  joint likelihood    : {human(table)}\n"
            f"  joint log posterior : {human(log_posterior)}\n"
            f"  peak (approx)       : {human(table + log_posterior)}"
        )

    def warn_if_large(self, batch_size: int | None = None, limit_gib: float = 2.0) -> None:
        """Emit a ``ResourceWarning`` when the joint table is likely to hurt."""
        shape = self.joint_likelihood_shape(batch_size)
        if int(np.prod(shape, dtype=np.int64)) * 8 > limit_gib * 1024**3:
            warnings.warn(f"joint likelihood is large:\n{self.memory_report(batch_size)}.", ResourceWarning, stacklevel=3)

    def rng(self, seed: int | np.random.Generator | None = None) -> np.random.Generator:
        """Build a ``Generator``, preferring an explicit ``seed`` over ``self.seed``."""
        if isinstance(seed, np.random.Generator):
            return seed
        return np.random.default_rng(self.seed if seed is None else seed)

    def replace(self, **changes: Any) -> CogGridConfig:
        """Return a copy with ``changes`` applied (validation re-runs).
        An automatic ``n_held_out_vars`` keeps the value it resolved to; pass ``n_held_out_vars=None`` to re-derive it."""
        return replace(self, **changes)
