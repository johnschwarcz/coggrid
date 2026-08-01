"""The five places you can swap in your own behaviour.

The original code exposed customisation through ``Env_Customization.py``: a set
of ``custom_*`` methods that returned ``False`` by default, and which the
framework called *before* each default implementation. If your override returned
anything truthy, the default was skipped. Overriding meant editing a file inside
the library, the contract was "set these attributes on ``self``", and nothing
told you what shape they should be.

Here the same five stages are explicit callables with typed signatures. You pass
your own to :class:`~coggrid.world.World`; you never edit the
library. Because each one returns its outputs rather than assigning to ``self``,
you can unit-test a replacement in isolation.

.. code-block:: python

    def blocky_embeddings(cfg, rng):
        "Embeddings drawn from a small discrete set instead of a Gaussian."
        codebook = rng.standard_normal((8, cfg.n_observations, cfg.embedding_dim))
        pick = rng.integers(0, 8, size=cfg.n_states)
        k = codebook[pick]
        k /= np.linalg.norm(k, axis=-1, keepdims=True)
        return k, k.copy()

    world = World(cfg, embeddings=blocky_embeddings)

Each protocol is runtime-checkable, so ``isinstance(f, EmbeddingSource)`` works
for a quick sanity check, but nothing forces you to subclass anything — a plain
function with the right signature is enough.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .config import CogGridConfig
from .generative import Split

__all__ = [
    "EmbeddingSource",
    "ContextSampler",
    "LikelihoodModel",
    "RealizationSampler",
    "ObservationModel",
]


@runtime_checkable
class EmbeddingSource(Protocol):
    """Produce the per-state key and query embeddings.

    Called once when a :class:`World` is built, not once per episode.
    """

    def __call__(
        self, cfg: CogGridConfig, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(keys, queries)``, each ``(n_states, n_observations, embedding_dim)``."""
        ...


@runtime_checkable
class ContextSampler(Protocol):
    """Choose which latent states are active, and which carries the goal."""

    def __call__(
        self,
        cfg: CogGridConfig,
        rng: np.random.Generator,
        split: Split,
        n_episodes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(ctx_inds, goal_ind)`` of shapes ``(n_eps, n_ctx)`` and ``(n_eps,)``."""
        ...


@runtime_checkable
class LikelihoodModel(Protocol):
    """Map active-state embeddings to Bernoulli rates for every joint realisation.

    Replace this to change the *structure* of the dependency between latent
    states — e.g. to add a three-way interaction term, or to make the
    interaction symmetric.
    """

    def __call__(
        self,
        cfg: CogGridConfig,
        keys: np.ndarray,
        queries: np.ndarray,
        ctx_inds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(rates, interactions)``.

        ``rates`` has shape ``(n_eps, n_obs, *(n_realizations,) * n_contexts)``
        and must lie strictly inside ``(0, 1)``. ``interactions`` is any
        ``(n_eps, n_obs, k)`` diagnostic array; it is carried along for analysis
        and never used by the observers.
        """
        ...


@runtime_checkable
class RealizationSampler(Protocol):
    """Draw the value each active state actually takes this episode."""

    def __call__(
        self,
        cfg: CogGridConfig,
        rng: np.random.Generator,
        goal_ind: np.ndarray,
        n_episodes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(ctx_vals, goal_value)`` of shapes ``(n_eps, n_ctx)`` and ``(n_eps,)``."""
        ...


@runtime_checkable
class ObservationModel(Protocol):
    """Turn the true joint realisation into an observation stream.

    Replace this to make observations non-stationary, correlated across time, or
    continuous rather than binary. Note that the built-in Bayesian observers
    assume i.i.d. Bernoulli observations, so a replacement here generally needs
    a matching observer.
    """

    def __call__(
        self,
        cfg: CogGridConfig,
        rates: np.ndarray,
        ctx_vals: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(true_rates, observations)``.

        ``true_rates`` is ``(n_eps, n_obs)``; ``observations`` is a boolean
        ``(n_eps, n_steps, n_obs)`` array.
        """
        ...
