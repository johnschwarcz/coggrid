"""Configuration for a CogGrid world.

Everything that used to be a loose ``init.get('...')`` lookup inside
``CognitiveGridworld.__init__`` now lives here as a typed, validated,
immutable dataclass. Nothing in this module touches global state, allocates
large arrays, or runs a simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

__all__ = ["CogGridConfig", "LEGACY_NAMES"]


#: Old keyword name -> new field name. Used by :meth:`CogGridConfig.from_legacy`
#: so that scripts written against the original API keep working.
LEGACY_NAMES: dict[str, str] = {
    "state_num": "n_states",
    "ctx_num": "n_contexts",
    "realization_num": "n_realizations",
    "obs_num": "n_observations",
    "step_num": "n_steps",
    "batch_num": "n_episodes",
    "KQ_dim": "embedding_dim",
    "test_states": "n_held_out_states",
    "subsample_states": "subsample_states",
    "likelihood_temp": "likelihood_temp",
    "likelihood_freq": "likelihood_freq",
}

#: Original keywords that only ever configured the (now removed) neural
#: networks. Passing one to :meth:`CogGridConfig.from_legacy` is ignored
#: with a warning rather than raising, so old dicts can be dropped in wholesale.
DROPPED_NAMES: frozenset[str] = frozenset({
    "episodes", "hid_dim", "mode", "training", "reservoir", "learn_embeddings",
    "classifier_LR", "controller_LR", "generator_LR", "classifier_ent_bonus",
    "control_ent_bonus", "cuda", "checkpoint_every", "plot_every", "show_plots",
    "showtime", "skip_training_analyses", "early_stopping", "load_env",
    "save_env", "load_warnings",
})


@dataclass(frozen=True, slots=True)
class CogGridConfig:
    """Immutable description of a CogGrid world.

    The world is a *stationary* POMDP. On each episode:

    1. ``n_contexts`` latent states are drawn from a pool of ``n_states``;
    2. each active state takes one of ``n_realizations`` discrete values;
    3. those values jointly determine a Bernoulli rate for each of
       ``n_observations`` binary observation channels;
    4. the agent sees ``n_steps`` i.i.d. samples of those channels and must
       infer the value of the single *goal* state.

    Because the Bernoulli rates come from **pairwise** interactions between
    active states, the joint observation distribution does not factorise over
    contexts. The gap between an observer that models the interactions and one
    that assumes independence is the quantity of interest.

    Attributes
    ----------
    n_states:
        Size of the latent state pool that contexts are drawn from.
    n_contexts:
        Number of simultaneously active latent states per episode. Note that
        the joint likelihood table is ``n_realizations ** n_contexts`` wide, so
        this is the parameter that drives memory use.
    n_realizations:
        Number of discrete values each active state can take.
    n_observations:
        Number of binary observation channels.
    n_steps:
        Observation samples per episode (the episode horizon).
    embedding_dim:
        Dimensionality of the per-state key/query embeddings. Must be at least
        ``n_observations``, since embeddings are orthogonalised across channels.
    likelihood_temp:
        Scales the interaction potentials before the sigmoid. Larger values push
        Bernoulli rates towards 0/1, making single observations more informative.
    likelihood_freq:
        Number of periods in the sinusoidal value profile. Higher values make
        the mapping from interaction strength to realisation multimodal.
    n_episodes:
        Default batch size for :meth:`~coggrid.World.sample_episodes`.
    n_held_out_states:
        Size of the held-out ("test") slice of the state pool. Defaults to
        ``n_states // 10``. Held-out states are ``range(n_held_out_states)``;
        training states are the remainder.
    subsample_states:
        If set, draw contexts from a random subset of this size within each
        split. Useful for probing generalisation with a fixed small support.
    allow_repeated_states:
        Whether a single episode may activate the same latent state twice. The
        original implementation always allowed this on the held-out split; set
        to ``False`` for distinct-state episodes.
    seed:
        Seed for the default RNG. ``None`` means non-reproducible.
    """

    n_states: int = 500
    n_contexts: int = 2
    n_realizations: int = 10
    n_observations: int = 5
    n_steps: int = 30
    embedding_dim: int = 30
    likelihood_temp: float = 2.0
    likelihood_freq: float = 1.0
    n_episodes: int = 1000
    n_held_out_states: int | None = None
    subsample_states: int | None = None
    allow_repeated_states: bool = True
    seed: int | None = None

    # ------------------------------------------------------------------ setup
    def __post_init__(self) -> None:
        positive = {
            "n_states": self.n_states,
            "n_contexts": self.n_contexts,
            "n_realizations": self.n_realizations,
            "n_observations": self.n_observations,
            "n_steps": self.n_steps,
            "embedding_dim": self.embedding_dim,
            "n_episodes": self.n_episodes,
        }
        for name, value in positive.items():
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")

        if self.embedding_dim < self.n_observations:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be >= n_observations "
                f"({self.n_observations}): embeddings are orthogonalised across "
                "observation channels, which is impossible in a lower-dimensional "
                "space."
            )
        if self.likelihood_temp <= 0:
            raise ValueError(f"likelihood_temp must be > 0, got {self.likelihood_temp}")

        held_out = self.n_held_out_states
        if held_out is None:
            object.__setattr__(self, "n_held_out_states", self.n_states // 10)
            held_out = self.n_held_out_states
        if not 0 <= held_out <= self.n_states:
            raise ValueError(
                f"n_held_out_states ({held_out}) must be in [0, n_states={self.n_states}]"
            )
        if held_out == 0:
            raise ValueError(
                "n_held_out_states resolved to 0 — there would be no held-out "
                "states to evaluate generalisation on. Increase n_states "
                "(>= 10) or set n_held_out_states explicitly."
            )
        if held_out == self.n_states:
            raise ValueError(
                "n_held_out_states equals n_states — there would be no training "
                "states left."
            )

        if self.subsample_states is not None:
            smallest = min(held_out, self.n_states - held_out)
            if not 1 <= self.subsample_states <= smallest:
                raise ValueError(
                    f"subsample_states ({self.subsample_states}) must be in "
                    f"[1, {smallest}] (the smaller of the two splits)"
                )

        if not self.allow_repeated_states and self.n_contexts > self._split_floor():
            raise ValueError(
                f"n_contexts ({self.n_contexts}) exceeds the number of distinct "
                f"states available in the smaller split ({self._split_floor()}), "
                "so distinct-state episodes are impossible. Either increase "
                "n_states / n_held_out_states or set allow_repeated_states=True."
            )

    def _split_floor(self) -> int:
        held_out = int(self.n_held_out_states or 0)
        if self.subsample_states is not None:
            return int(self.subsample_states)
        return min(held_out, self.n_states - held_out)

    # ------------------------------------------------------- derived quantities
    @property
    def n_train_states(self) -> int:
        """Number of latent states reserved for training episodes."""
        return self.n_states - int(self.n_held_out_states or 0)

    @property
    def n_roll(self) -> int:
        """Length of the circular value profile (``1 + 2 * n_realizations``).

        The profile is longer than ``n_realizations`` so that interaction
        strengths can push probability mass "off the end" of the realisation
        axis and wrap around, rather than piling up at the boundary.
        """
        return 1 + 2 * self.n_realizations

    @property
    def n_interactions(self) -> int:
        """Number of ordered active-state pairs whose interaction is recorded."""
        return max(1, self.n_contexts * (self.n_contexts - 1))

    @property
    def realization_shape(self) -> tuple[int, ...]:
        """Shape of the joint realisation axes: ``(n_realizations,) * n_contexts``."""
        return (self.n_realizations,) * self.n_contexts

    def joint_likelihood_shape(self, n_episodes: int | None = None) -> tuple[int, ...]:
        """Shape of the joint likelihood table for a batch."""
        n = self.n_episodes if n_episodes is None else n_episodes
        return (n, self.n_observations, *self.realization_shape)

    def joint_likelihood_nbytes(self, n_episodes: int | None = None) -> int:
        """Size in bytes of one float64 joint likelihood table."""
        shape = self.joint_likelihood_shape(n_episodes)
        return int(np.prod(shape, dtype=np.int64)) * 8

    def memory_report(self, n_episodes: int | None = None) -> str:
        """Human-readable estimate of the dominant allocation.

        ``n_realizations ** n_contexts`` grows fast; this is the number people
        need in front of them *before* they wait on an OOM.
        """
        n = self.n_episodes if n_episodes is None else n_episodes
        table = self.joint_likelihood_nbytes(n)
        belief = table * self.n_steps / self.n_observations
        total = table + belief

        def human(x: float) -> str:
            for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
                if x < 1024 or unit == "TiB":
                    return f"{x:.1f} {unit}"
                x /= 1024
            raise AssertionError  # pragma: no cover

        return (
            f"{n} episodes x {self.n_observations} channels x "
            f"{self.n_realizations}^{self.n_contexts} realisations\n"
            f"  joint likelihood : {human(table)}\n"
            f"  joint belief     : {human(belief)}\n"
            f"  peak (approx)    : {human(total)}"
        )

    def warn_if_large(self, n_episodes: int | None = None, limit_gib: float = 2.0) -> None:
        """Emit a ``ResourceWarning`` when the joint table is likely to hurt."""
        import warnings

        nbytes = self.joint_likelihood_nbytes(n_episodes)
        if nbytes > limit_gib * 1024**3:
            warnings.warn(
                "CogGrid joint likelihood is large:\n"
                + self.memory_report(n_episodes)
                + "\nReduce n_episodes, n_contexts or n_realizations, or sample "
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

    def replace(self, **changes: Any) -> "CogGridConfig":
        """Return a copy with ``changes`` applied (validation re-runs)."""
        return replace(self, **changes)

    @classmethod
    def from_legacy(cls, mapping: dict[str, Any] | None = None, /, **kwargs: Any) -> "CogGridConfig":
        """Build a config from the original ``CognitiveGridworld(**init)`` keywords.

        Renamed keys are translated; keys that only configured the removed
        neural networks are dropped with a warning.

        >>> CogGridConfig.from_legacy(state_num=100, ctx_num=2, obs_num=3).n_states
        100
        """
        import warnings

        merged: dict[str, Any] = {**(mapping or {}), **kwargs}
        known = {f.name for f in fields(cls)}
        out: dict[str, Any] = {}
        dropped: list[str] = []
        unknown: list[str] = []

        for key, value in merged.items():
            if key in DROPPED_NAMES:
                dropped.append(key)
            elif key in LEGACY_NAMES:
                out[LEGACY_NAMES[key]] = value
            elif key in known:
                out[key] = value
            else:
                unknown.append(key)

        if dropped:
            warnings.warn(
                "Ignoring keyword(s) that configured the removed network/training "
                f"code: {sorted(dropped)}. This package models the environment "
                "only.",
                UserWarning,
                stacklevel=2,
            )
        if unknown:
            raise TypeError(
                f"Unrecognised config keyword(s): {sorted(unknown)}. "
                f"Valid names: {sorted(known)}"
            )
        return cls(**out)

    def summary(self) -> str:
        """One-line-per-field description, for logging and provenance."""
        lines = [f"CogGridConfig("]
        for f in fields(self):
            lines.append(f"    {f.name}={getattr(self, f.name)!r},")
        lines.append(")")
        return "\n".join(lines)
