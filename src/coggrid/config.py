"""Every tunable of a CogGrid world, in one validated, immutable dataclass.

Nothing in this module touches global state, allocates large arrays, or runs a
simulation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

__all__ = ["CogGridConfig"]


@dataclass(frozen=True, slots=True)
class CogGridConfig:
    """Immutable description of a CogGrid world.

    The world is a *stationary* POMDP. On each episode:

    1. ``n_contexts`` latent variables are drawn from a pool of ``n_vars``;
    2. each active variable takes one of ``n_realizations`` discrete values;
    3. those values jointly determine a Bernoulli rate for each of
       ``n_observations`` binary observation channels;
    4. the agent sees ``n_steps`` i.i.d. samples of those channels and must
       infer the value of the single *goal* variable.

    Because the Bernoulli rates come from **pairwise** interactions between
    active variables, the joint observation distribution does not factorise over
    contexts. The gap between an observer that models the interactions and one
    that assumes independence is the quantity of interest.

    Attributes
    ----------
    n_vars:
        Size of the latent variable pool that contexts are drawn from.
    n_contexts:
        Number of simultaneously active latent variables per episode. Note that
        the joint likelihood table is ``n_realizations ** n_contexts`` wide, so
        this is the parameter that drives memory use.
    n_realizations:
        Number of discrete values each active variable can take.
    n_observations:
        Number of binary observation channels.
    n_steps:
        Observation samples per episode (the episode horizon).
    embedding_dim:
        Dimensionality of the per-variable key/query embeddings. Must be at least
        ``n_observations``, since embeddings are orthogonalized across channels.
    likelihood_temp:
        Scales the interaction potentials before the sigmoid. Larger values push
        Bernoulli rates towards 0/1, making single observations more informative.
    likelihood_freq:
        Number of periods in the sinusoidal value profile. Higher values make
        the mapping from interaction strength to realization multimodal.
    batch_size:
        Default batch size for :meth:`~coggrid.World.sample_episodes`.
    n_held_out_vars:
        Size of the held-out ("test") slice of the variable pool. ``None`` means
        ``n_vars // 10``, resolved to an int during validation. Held-out
        variables are ``range(n_held_out_vars)``; training variables are the rest.
    subsample_vars:
        If set, draw contexts from a random subset of this size within each
        split. Useful for probing generalization with a fixed small support.
    allow_repeated_vars:
        Whether a single episode may activate the same latent variable twice. Set
        to ``False`` for distinct-variable episodes.
    seed:
        Seed for the default RNG. ``None`` means non-reproducible.
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
    subsample_vars: int | None = None
    allow_repeated_vars: bool = True
    seed: int | None = None

    # ------------------------------------------------------------------ setup
    def __post_init__(self) -> None:
        positive = {
            "n_vars": self.n_vars,
            "n_contexts": self.n_contexts,
            "n_realizations": self.n_realizations,
            "n_observations": self.n_observations,
            "n_steps": self.n_steps,
            "embedding_dim": self.embedding_dim,
            "batch_size": self.batch_size,
        }
        for name, value in positive.items():
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")

        if self.embedding_dim < self.n_observations:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be >= n_observations "
                f"({self.n_observations}): embeddings are orthogonalized across "
                "observation channels, which is impossible in a lower-dimensional "
                "space."
            )
        if self.likelihood_temp <= 0:
            raise ValueError(f"likelihood_temp must be > 0, got {self.likelihood_temp}")

        # Resolve the "auto" default once, so every reader downstream sees an int.
        held_out = self.n_held_out_vars
        if held_out is None:
            held_out = self.n_vars // 10
        object.__setattr__(self, "n_held_out_vars", held_out)
        if not 0 <= held_out <= self.n_vars:
            raise ValueError(
                f"n_held_out_vars ({held_out}) must be in [0, n_vars={self.n_vars}]"
            )
        if held_out == 0:
            raise ValueError(
                "n_held_out_vars resolved to 0 — there would be no held-out "
                "variables to evaluate generalization on. Increase n_vars "
                "(>= 10) or set n_held_out_vars explicitly."
            )
        if held_out == self.n_vars:
            raise ValueError(
                "n_held_out_vars equals n_vars — there would be no training "
                "variables left."
            )

        if self.subsample_vars is not None:
            smallest = min(held_out, self.n_vars - held_out)
            if not 1 <= self.subsample_vars <= smallest:
                raise ValueError(
                    f"subsample_vars ({self.subsample_vars}) must be in "
                    f"[1, {smallest}] (the smaller of the two splits)"
                )

        if not self.allow_repeated_vars and self.n_contexts > self._split_floor():
            raise ValueError(
                f"n_contexts ({self.n_contexts}) exceeds the number of distinct "
                f"variables available in the smaller split ({self._split_floor()}), "
                "so distinct-variable episodes are impossible. Either increase "
                "n_vars / n_held_out_vars or set allow_repeated_vars=True."
            )

    def _split_floor(self) -> int:
        """Distinct variables available in the smaller of the two splits."""
        if self.subsample_vars is not None:
            return self.subsample_vars
        return min(self.n_held_out_vars, self.n_vars - self.n_held_out_vars)

    # ------------------------------------------------------- derived quantities
    @property
    def n_train_vars(self) -> int:
        """Number of latent variables reserved for training episodes."""
        return self.n_vars - self.n_held_out_vars

    @property
    def n_roll(self) -> int:
        """Length of the circular value profile (``1 + 2 * n_realizations``).

        The profile is longer than ``n_realizations`` so that interaction
        strengths can push probability mass "off the end" of the realization
        axis and wrap around, rather than piling up at the boundary.
        """
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
        """Human-readable estimate of the dominant allocation.

        ``n_realizations ** n_contexts`` grows fast; this is the number people
        need in front of them *before* they wait on an OOM.

        Examples
        --------
        >>> print(CogGridConfig(n_contexts=4, n_realizations=20).memory_report(1000))
        1000 episodes x 5 channels x 20^4 realizations
          joint likelihood : 6.0 GiB
          joint belief     : 35.8 GiB
          peak (approx)    : 41.7 GiB
        """
        n = self.batch_size if batch_size is None else batch_size
        table = int(np.prod(self.joint_likelihood_shape(n), dtype=np.int64)) * 8
        belief = table * self.n_steps / self.n_observations

        def human(x: float) -> str:
            for unit in ("B", "KiB", "MiB", "GiB"):
                if x < 1024:
                    return f"{x:.1f} {unit}"
                x /= 1024
            return f"{x:.1f} TiB"

        return (
            f"{n} episodes x {self.n_observations} channels x "
            f"{self.n_realizations}^{self.n_contexts} realizations\n"
            f"  joint likelihood : {human(table)}\n"
            f"  joint belief     : {human(belief)}\n"
            f"  peak (approx)    : {human(table + belief)}"
        )

    def warn_if_large(
        self, batch_size: int | None = None, limit_gib: float = 2.0
    ) -> None:
        """Emit a ``ResourceWarning`` when the joint table is likely to hurt."""
        shape = self.joint_likelihood_shape(batch_size)
        if int(np.prod(shape, dtype=np.int64)) * 8 > limit_gib * 1024**3:
            warnings.warn(
                "CogGrid joint likelihood is large:\n"
                + self.memory_report(batch_size)
                + "\nReduce batch_size, n_contexts or n_realizations, or sample "
                "in chunks.",
                ResourceWarning,
                stacklevel=3,
            )

    # ---------------------------------------------------------------- helpers
    def rng(self, seed: int | np.random.Generator | None = None) -> np.random.Generator:
        """Build a ``Generator``, preferring an explicit ``seed`` over ``self.seed``."""
        if isinstance(seed, np.random.Generator):
            return seed
        return np.random.default_rng(self.seed if seed is None else seed)

    def replace(self, **changes: Any) -> CogGridConfig:
        """Return a copy with ``changes`` applied (validation re-runs)."""
        return replace(self, **changes)
