"""coggrid — a stationary POMDP for studying compositional generalization
in latent space.

The "grid" is the table of observation likelihoods over the joint realizations
of the active latent variables.

Quick start
-----------
>>> from coggrid import CogGridConfig, World, run_observers
>>> world = World(CogGridConfig(n_vars=200, n_contexts=2, seed=0))
>>> batch = world.sample_episodes(256)
>>> traces = run_observers(batch)
>>> traces["joint"].final()["accuracy"] > traces["naive"].final()["accuracy"]
True

Gymnasium interface
-------------------
>>> from coggrid import CogGridEnv
>>> env = CogGridEnv(seed=0)
>>> obs, info = env.reset()
>>> obs, reward, terminated, truncated, info = env.step(0)

Layout
------
``config``       :class:`CogGridConfig` — every tunable, validated, immutable.
``sampling``     The random draws: contexts, realizations, observations.
``likelihood``   How realizations set ``P(obs = 1)``: embeddings, sinusoids, the likelihood tables.
``world``        :class:`World` and :class:`EpisodeBatch`.
``observers``    Ideal-observer baselines and metrics.
``env``          :class:`CogGridEnv`, the Gymnasium environment.
``torch``        Optional CUDA acceleration of the joint observer.
``viz``          Plotting and animation; nothing calls ``show()``.
"""

from __future__ import annotations

from .config import CogGridConfig
from .env import CogGridEnv
from .observers import (
    BeliefTrace,
    disentanglement,
    factorization_regret,
    joint_observer,
    naive_observer,
    run_observers,
    score_belief,
)
from .world import EmbeddingSource, EpisodeBatch, World

__version__ = "0.3.0"

__all__ = [
    "__version__",
    # configuration
    "CogGridConfig",
    # environment
    "World",
    "EpisodeBatch",
    "CogGridEnv",
    # observers
    "BeliefTrace",
    "joint_observer",
    "naive_observer",
    "run_observers",
    "score_belief",
    "factorization_regret",
    "disentanglement",
    # extension point
    "EmbeddingSource",
]
